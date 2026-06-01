import glob
import logging
import os
import queue
import shutil
import time
import numpy as np
import torch
import torch.multiprocessing as _mp
from torch import optim
from dataset.dataset import get_wm811k, get_wm811k_loaders
from model.model import ResnetModel
from utils.trainer_pbt_v2 import PBT
from utils.utils import format_time, set_seed, MultiTaskLossWeighting, FocalLoss, exploit_and_explore
from torch.utils.data import DataLoader, SequentialSampler

CHECKPOINT_DIR = "./checkpoints"
POPULATION_SIZE = 3
NUM_WORKERS = 3
BATCH_SIZE = 256
MU = 4
MAX_PBT_ROUNDS = 10 #30
PBT_INTERVAL = 2 #5
LABEL_RATIO = 1.00
INIT_HYPERPARAMETERS = {
    "gamma": (1.0, 1.75, 3.5),
    "threshold": (0.90, 0.925, 0.95),
}
EXPLOIT_FRACTION = 0.2

mp = _mp.get_context("spawn")
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def population_checkpoint_path(checkpoints_dir, task_id):
    return os.path.join(checkpoints_dir, f"task-{task_id:02d}_population.pth")

def best_checkpoint_path(checkpoints_dir):
    return os.path.join(checkpoints_dir, "best_pbt.pth")

def best_metadata_path(checkpoints_dir):
    return os.path.join(checkpoints_dir, "best_pbt_meta.pt")

def reset_task_checkpoints(checkpoints_dir):
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for ckpt_file in glob.glob(os.path.join(checkpoints_dir, "task-*.pth")):
        os.remove(ckpt_file)

    for ckpt_file in (best_checkpoint_path(checkpoints_dir), best_metadata_path(checkpoints_dir)):
        if os.path.isfile(ckpt_file):
            os.remove(ckpt_file)





def initialize_population(population_q, population_size, init_hyperparameter, checkpoints_dir):
    assert population_size == len(init_hyperparameter["gamma"]), \
        "population_size must match the number of gamma candidates."
    assert population_size == len(init_hyperparameter["threshold"]), \
        "population_size must match the number of threshold candidates."
    
    logs = ["[PBT Initial]"]
    for task_id, (gamma, threshold) in enumerate(zip(init_hyperparameter["gamma"], init_hyperparameter["threshold"])):
        
        population_q.put({"id": task_id,
                          "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                          "alpha": 1.0, "beta": 1.0, # init alpha and beta for MTL weighting
                          "gamma": float(gamma), "threshold": float(threshold),
                          "gamma_history": [float(gamma)], "threshold_history": [float(threshold)],
                          "alpha_history": [], "beta_history": []
                          })

        model = ResnetModel(model_name="resnet18", num_classes=9, pretrained=True).to(torch.device("cpu"))
        mtl_weighting = MultiTaskLossWeighting(num=2).to(torch.device("cpu"))
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            [{"params": model_params}, {"params": mtl_weighting.parameters()}],
            lr=1e-3,
            weight_decay=1e-5,
        )

        torch.save({"model_state_dict": model.state_dict(),
                    "mtl_state_dict": mtl_weighting.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "gamma": float(gamma),
                    "threshold": float(threshold)}, population_checkpoint_path(checkpoints_dir, task_id))
        logs.append(f"task {task_id}: gamma={float(gamma):.3f}, threshold={float(threshold):.3f}, alpha=1, beta=1")

    print("\n".join(logs))
    return init_hyperparameter["gamma"], init_hyperparameter["threshold"]


class Worker(mp.Process):
    def __init__(self, train_labeled_dataset, train_unlabeled_dataset, 
                 val_dataset, test_dataset, batch_size, mu,
                 pbt_interval, round_counter, max_pbt_rounds, population_q, 
                 finish_tasks_q, checkpoints_dir, best_score, best_lock, device):
        super().__init__()
        
        # dataset
        self.train_labeled_dataset = train_labeled_dataset
        self.train_unlabeled_dataset = train_unlabeled_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # loader * model
        self.batch_size = batch_size
        self.mu = mu
        self.device = device
        
        # pbt
        self.population_q = population_q # queue for population of tasks to train
        self.finish_tasks_q = finish_tasks_q # queue for finished tasks
        
        self.pbt_interval = pbt_interval
        self.round_counter = round_counter
        self.max_pbt_rounds = max_pbt_rounds

        self.checkpoints_dir = checkpoints_dir
        self.best_score = best_score
        self.best_lock = best_lock

    def build_trainer(self):
        labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_wm811k_loaders(self.train_labeled_dataset, self.train_unlabeled_dataset, self.val_dataset, self.test_dataset, self.batch_size, self.mu)
        model = ResnetModel(model_name="resnet18", num_classes=9, pretrained=True).to(self.device)
        mtl_weighting = MultiTaskLossWeighting(num=2).to(self.device)
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam([{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": 1e-3, "weight_decay": 1e-5},
                                {"params": mtl_weighting.parameters(), "lr": 1e-4, "weight_decay": 0.0}
                                ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.max_pbt_rounds * self.pbt_interval))
        focal_loss = FocalLoss(gamma=2.5, alpha=1.0, reduction="none")

        return PBT(model=model, labeled_trainloader=labeled_trainloader, unlabeled_trainloader=unlabeled_trainloader,
            val_loader=val_loader, test_loader=test_loader,
            epochs=self.pbt_interval, optimizer=optimizer,
            scheduler=scheduler, early_stopping=None, 
            temperature=1.0, threshold=0.90, focal_loss=focal_loss, mtl_weighting=mtl_weighting,
            use_amp=True, device=self.device)

    def run(self):
        trainer = self.build_trainer()

        while self.round_counter.value < self.max_pbt_rounds:
            try:
                task_dict = self.population_q.get(timeout=5) # get a task from the population queue
                # task format = {"id", "gamma", "accuracy", "precision", "recall", "f1_score", "threshold", "gamma_history", "threshold_history"}
            except queue.Empty:
                continue

            if self.round_counter.value >= self.max_pbt_rounds:
                self.population_q.put(task_dict)
                break

            
            # Load checkpoint and hyperparameters for the task
            trainer.set_id(task_id = task_dict["id"])
            task_checkpoint_path = population_checkpoint_path(self.checkpoints_dir, trainer.task_id)
            trainer.load_checkpoint(task_checkpoint_path)
            
            best_interval_score = -1.0
            best_interval_metrics = None
            best_interval_train_metrics = None
            alpha_epoch_history = list(task_dict.get("alpha_history", []))
            beta_epoch_history = list(task_dict.get("beta_history", []))

            for local_epoch in range(self.pbt_interval):
                _, train_metrics = trainer.train_one_epoch(local_epoch)
                _, metrics = trainer.evaluate(data_loader=trainer.val_loader, loader_name="Validation Evaluating...")
                alpha_epoch_history.append(float(train_metrics.get("alpha", 0.0)))
                beta_epoch_history.append(float(train_metrics.get("beta", 0.0)))

                global_epoch = self.round_counter.value * self.pbt_interval + local_epoch + 1
                current_f1 = float(metrics["f1"])

                if current_f1 > best_interval_score:
                    best_interval_score = current_f1
                    best_interval_metrics = dict(metrics)
                    best_interval_train_metrics = dict(train_metrics)
                    trainer.save_checkpoint(task_checkpoint_path)

                    self.save_global_checkpoint(
                        task_dict=task_dict,
                        task_id=trainer.task_id,
                        metrics=metrics,
                        train_metrics=train_metrics,
                        alpha_history=alpha_epoch_history,
                        beta_history=beta_epoch_history,
                        round_idx=self.round_counter.value + 1,
                        global_epoch=global_epoch,
                        task_checkpoint_path=task_checkpoint_path,
                    )

            metrics = best_interval_metrics
            train_metrics = best_interval_train_metrics if best_interval_train_metrics is not None else {}
            self.finish_tasks_q.put({
                        "id": trainer.task_id,
                        "accuracy": float(metrics["accuracy"]), "precision": float(metrics["precision"]), "recall": float(metrics["recall"]), "f1_score": float(metrics["f1"]),
                        "alpha": float(train_metrics.get("alpha", 0.0)), 
                        "beta": float(train_metrics.get("beta", 0.0)),
                        "gamma": float(getattr(trainer.focal_loss, "gamma", 2.5)), 
                        "threshold": float(trainer.threshold),
                        "gamma_history": task_dict.get("gamma_history", []),
                        "threshold_history": task_dict.get("threshold_history", []),
                        "alpha_history": alpha_epoch_history,
                        "beta_history": beta_epoch_history
                        })
        self.finish_tasks_q.put({"type": "DONE", "worker": self.name})

    def save_global_checkpoint(self, task_dict, task_id, metrics, train_metrics, alpha_history, beta_history, global_epoch, round_idx, task_checkpoint_path):
        with self.best_lock:
            if float(metrics["f1"]) <= self.best_score.value:
                return

            self.best_score.value = float(metrics["f1"])

            shutil.copyfile(task_checkpoint_path, best_checkpoint_path(self.checkpoints_dir))
            torch.save({"id": int(task_id), "accuracy": float(metrics["accuracy"]), "precision": float(metrics["precision"]), "recall": float(metrics["recall"]), "f1_score": float(metrics["f1"]),
                    "alpha": float(train_metrics.get("alpha", 0.0)), "beta": float(train_metrics.get("beta", 0.0)),
                    "gamma": float(self._get_task_value(task_dict, "gamma")), "threshold": float(self._get_task_value(task_dict, "threshold")),
                    "gamma_history": [float(x) for x in task_dict.get("gamma_history", [])], "threshold_history": [float(x) for x in task_dict.get("threshold_history", [])],
                    "alpha_history": [float(x) for x in alpha_history],
                    "beta_history": [float(x) for x in beta_history],
                    "round": int(round_idx), "epoch": int(global_epoch)}, best_metadata_path(self.checkpoints_dir))

    def _get_task_value(self, task_dict, key):
        return task_dict.get(key, 0.0)



class Explorer(mp.Process):
    def __init__(self, round_counter, max_pbt_rounds, population_q, finish_tasks_q, checkpoints_dir, exploit_fraction, population_size, num_workers):
        super().__init__()
        self.round_counter = round_counter
        self.max_pbt_rounds = max_pbt_rounds
        self.population_q = population_q
        self.finish_tasks_q = finish_tasks_q
        self.checkpoints_dir = checkpoints_dir
        self.exploit_fraction = exploit_fraction
        self.population_size = population_size
        self.num_workers = num_workers

    def run(self):  
        done_workers = set()
        while True:
            if len(done_workers) >= self.num_workers:
                break
            tasks = []
            while len(tasks) < self.population_size:
                try:
                    item = self.finish_tasks_q.get(timeout=5)
                except queue.Empty:
                    if len(done_workers) >= self.num_workers:
                        break
                    continue

                if isinstance(item, dict) and item.get("type") == "DONE":
                    done_workers.add(item.get("worker"))
                    continue

                tasks.append(item)

            if len(tasks) < self.population_size:
                if self.round_counter.value >= self.max_pbt_rounds:
                    continue
                continue

            tasks = sorted(tasks, key=lambda x: x["f1_score"], reverse=True)
            cutoff = max(1, int(np.ceil(self.exploit_fraction * len(tasks))))
            tops = tasks[:cutoff]
            bottoms = tasks[-cutoff:]

            round_idx = self.round_counter.value + 1
            updates_log = []
            for bottom in bottoms:
                top = tops[np.random.randint(len(tops))]
                updated_gamma, updated_threshold = exploit_and_explore(
                    population_checkpoint_path(self.checkpoints_dir, top["id"]),
                    population_checkpoint_path(self.checkpoints_dir, bottom["id"]),
                )
                bottom["gamma"] = float(updated_gamma)
                bottom["threshold"] = float(updated_threshold)
                bottom.setdefault("gamma_history", []).append(float(updated_gamma))
                bottom.setdefault("threshold_history", []).append(float(updated_threshold))
                updates_log.append(f"{bottom['id']}<-{top['id']}(gamma={float(updated_gamma):.4f}, threshold={float(updated_threshold):.4f})")

            current_epoch = round_idx * PBT_INTERVAL
            total_epoch = self.max_pbt_rounds * PBT_INTERVAL
            print(f"\n[PBT Round {round_idx}/{self.max_pbt_rounds}, epoch {current_epoch}/{total_epoch}]")
            print(
                f"best={tasks[0]['id']} | "
                f"(gamma={tasks[0].get('gamma', 0.0):.4f}, threshold={tasks[0].get('threshold', 0.0):.4f}, alpha={tasks[0].get('alpha', 0.0):.4f}, beta={tasks[0].get('beta', 0.0):.4f}) | "
                f"accuracy={tasks[0]['accuracy']:.4f}, precision={tasks[0]['precision']:.4f}, recall={tasks[0]['recall']:.4f}, f1-score={tasks[0]['f1_score']:.4f}"
              
            )
            print(
                f"worst={tasks[-1]['id']} | "
                f"(gamma={tasks[-1].get('gamma', 0.0):.4f}, threshold={tasks[-1].get('threshold', 0.0):.4f}, alpha={tasks[-1].get('alpha', 0.0):.4f}, beta={tasks[-1].get('beta', 0.0):.4f}) | "
                f"accuracy={tasks[-1]['accuracy']:.4f}, precision={tasks[-1]['precision']:.4f}, recall={tasks[-1]['recall']:.4f}, f1-score={tasks[-1]['f1_score']:.4f}"
            )
            print(f"update={', '.join(updates_log)}")

            with self.round_counter.get_lock():
                self.round_counter.value += 1

            for task in tasks:
                self.population_q.put(task)


if __name__ == "__main__":
    set_seed(42)
    reset_task_checkpoints(CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_wm811k(
        labeled_path="./data/wm811k/preprocessing/labeled.pkl", unlabeled_path="./data/wm811k/preprocessing/unlabeled.pkl",
        train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, label_ratio=LABEL_RATIO, image_size=96,
        cutout_num_holes=4, cutout_ratio=0.2, noise_prob=0.05,
        data_seed=0)

    logger.info(f"train_labeled_dataset: {len(train_labeled_dataset)}")
    logger.info(f"val_dataset: {len(val_dataset)}")
    logger.info(f"test_dataset: {len(test_dataset)}")
    logger.info(f"train_unlabeled_dataset: {len(train_unlabeled_dataset)}")


    population_q = mp.Queue(maxsize=POPULATION_SIZE) # Queue that stores tasks waiting to be trained by Worker processes.
    finish_tasks_q = mp.Queue(maxsize=POPULATION_SIZE) # Queue that stores tasks after Workers finish training and evaluation.
    round_counter = mp.Value("i", 0) # Shared counter for the number of completed PBT rounds.
    best_score = mp.Value("d", -1.0) # Shared variable that stores the best validation score found by all Workers.
    best_lock = mp.Lock() # Lock used to prevent multiple Workers from updating the best score and best checkpoint files at the same time.



    init_gammas, init_thresholds = initialize_population(population_q=population_q, population_size=POPULATION_SIZE, 
                                                         init_hyperparameter=INIT_HYPERPARAMETERS, checkpoints_dir=CHECKPOINT_DIR)

    
    workers = [Worker(
                    train_labeled_dataset=train_labeled_dataset,
                    train_unlabeled_dataset=train_unlabeled_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    batch_size=BATCH_SIZE,
                    mu=MU,
                    pbt_interval=PBT_INTERVAL,
                    round_counter=round_counter,
                    max_pbt_rounds=MAX_PBT_ROUNDS,
                    population_q=population_q,
                    finish_tasks_q=finish_tasks_q,
                    checkpoints_dir=CHECKPOINT_DIR,
                    best_score=best_score,
                    best_lock=best_lock,
                    device=device) for _ in range(NUM_WORKERS)
                ]
    for i, worker in enumerate(workers):
        worker.name = f"Worker-{i}"
    
    explorer = Explorer(
                    round_counter=round_counter,
                    max_pbt_rounds=MAX_PBT_ROUNDS,
                    population_q=population_q,
                    finish_tasks_q=finish_tasks_q,
                    checkpoints_dir=CHECKPOINT_DIR,
                    exploit_fraction=EXPLOIT_FRACTION,
                    population_size=POPULATION_SIZE,
                    num_workers=NUM_WORKERS)
    explorer.name = "Explorer"
    workers.append(explorer)
    total_start = time.perf_counter()

    for worker in workers:
        worker.start()
        
    for worker in workers:
        worker.join()

    tasks = []
    
    while not finish_tasks_q.empty():
        item = finish_tasks_q.get()
        if isinstance(item, dict) and item.get("type") == "DONE":
            continue
        tasks.append(item)
    while not population_q.empty():
        tasks.append(population_q.get())
    
    total_time = time.perf_counter() - total_start
    
    # Print final results
    tasks = sorted(tasks, key=lambda x: x["f1_score"], reverse=True)
    best_meta = torch.load(best_metadata_path(CHECKPOINT_DIR), map_location="cpu", weights_only=True)
    best_ckpt = torch.load(best_checkpoint_path(CHECKPOINT_DIR), map_location="cpu", weights_only=True)
    total_epochs_per_task = MAX_PBT_ROUNDS * PBT_INTERVAL
    total_task_epochs = POPULATION_SIZE * total_epochs_per_task
    
    print(f"\nTuning time: {format_time(total_time)}\n")
    print("[PBT Result]")
    print(f"Best task={best_meta['id']} | round={best_meta['round']} | epoch={best_meta['epoch']}")
    print(f"metrics: accuracy={best_meta['accuracy']:.4f}, precision={best_meta['precision']:.4f}, recall={best_meta['recall']:.4f}, f1-score={best_meta['f1_score']:.4f}")
    print(f"hyperparams: gamma={best_ckpt.get('gamma'):.4f}, threshold={best_ckpt.get('threshold', 0.0):.4f}, alpha={best_meta.get('alpha', 0.0):.4f}, beta={best_meta.get('beta', 0.0):.4f}")
    
    print("Gamma history:", [round(x, 4) for x in best_meta.get("gamma_history", [])])
    print("Threshold history:", [round(x, 4) for x in best_meta.get("threshold_history", [])])
    print("Alpha history:", [round(x, 4) for x in best_meta.get("alpha_history", [])])
    print("Beta history:", [round(x, 4) for x in best_meta.get("beta_history", [])])
    print(f"Epochs: {total_epochs_per_task} per task | {total_task_epochs} task-epochs")
    


    # Test the best model on the test data
    val_loader = DataLoader(val_dataset,sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE, drop_last=False)
    test_loader = DataLoader(test_dataset,sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE, drop_last=False)
    best_model = ResnetModel(model_name="resnet18", num_classes=9, pretrained=False).to(device)
    best_mtl_weighting = MultiTaskLossWeighting(num=2).to(device)
    best_model_params = [p for p in best_model.parameters() if p.requires_grad]
    best_optimizer = optim.Adam(
        [{"params": best_model_params}, {"params": best_mtl_weighting.parameters()}],
        lr=1e-4,
        weight_decay=1e-5,
    )

    best_trainer = PBT(model=best_model, labeled_trainloader=None, unlabeled_trainloader=None, val_loader=val_loader,
        test_loader=test_loader, epochs=0, optimizer=best_optimizer,
        scheduler=None, early_stopping=None, temperature=1.0, threshold=0.90,
        focal_loss=FocalLoss(gamma=2.5, alpha=1.0, reduction="none"),
        mtl_weighting=best_mtl_weighting,
        use_amp=True, device=device)
    best_trainer.load_checkpoint(best_checkpoint_path(CHECKPOINT_DIR))
    best_trainer._evaluate_and_log(best_trainer.val_loader, "Best PBT Validation")
    best_trainer._evaluate_and_log(best_trainer.test_loader, "Best PBT Test")
