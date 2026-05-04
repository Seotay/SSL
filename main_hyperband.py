import logging
import torch
import optuna

from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataset.dataset import get_wm811k
from model.model import ResnetModel
from utils.trainer import HyperbandTrainer
from utils.utils import EarlyStopping, set_seed


logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def build_dataloaders(label_ratio=1.00, batch_size=256, mu=4, data_seed=0):
    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_wm811k(
        labeled_path="./data/wm811k/preprocessing/labeled.pkl",
        unlabeled_path="./data/wm811k/preprocessing/unlabeled.pkl",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        image_size=96,
        data_seed=data_seed,
        label_ratio=label_ratio,
        cutout_num_holes=4,
        cutout_ratio=0.2,
        noise_prob=0.05,
    )

    logger.info(f"train_labeled_dataset: {len(train_labeled_dataset)}")
    logger.info(f"val_dataset: {len(val_dataset)}")
    logger.info(f"test_dataset: {len(test_dataset)}")
    logger.info(f"train_unlabeled_dataset: {len(train_unlabeled_dataset)}")

    labeled_trainloader = DataLoader(train_labeled_dataset,
                                     sampler=RandomSampler(train_labeled_dataset),
                                     batch_size=batch_size,
                                     num_workers=6, pin_memory=True, persistent_workers=True, drop_last=False)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset,
                                       sampler=RandomSampler(train_unlabeled_dataset),
                                       batch_size=batch_size * mu,
                                       num_workers=6, pin_memory=True, persistent_workers=True, drop_last=False)
    val_loader = DataLoader(val_dataset,
                            sampler=SequentialSampler(val_dataset),
                            batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=batch_size, drop_last=False)
    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader

def build_model(device):
    model = ResnetModel(model_name="resnet50", num_classes=9, pretrained=True).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.backbone.fc.parameters():
        param.requires_grad = True
    return model



def objective(trial, data_loaders, params, device):
    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = data_loaders
    model = build_model(device)

    early_stopping = EarlyStopping(patience=params["patience"], verbose=True, delta=0.0, path=f"./checkpoints/best_model_trial_{trial.number}.pt")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    lambda_u = trial.suggest_float("lambda_u", 0.1, 3.0)
    threshold = trial.suggest_float("threshold", 0.85, 0.975)

    trainer = HyperbandTrainer(
        model=model, labeled_trainloader=labeled_trainloader, unlabeled_trainloader=unlabeled_trainloader, val_loader=val_loader, test_loader=test_loader,
        epochs=params["epochs"], optimizer=optimizer, scheduler=scheduler, early_stopping=early_stopping, 
        lambda_u=lambda_u, temperature=params["temperature"], threshold=threshold,
        use_amp=True, device=device, 
        trial=trial)

    best_val_metrics = trainer.training()
    return best_val_metrics["f1"]


def evaluate_best_trial(study, dataloaders, device):
    labeled_trainloader, _, val_loader, test_loader = dataloaders
    model = build_model(device)
    best_trial_number = study.best_trial.number
    checkpoint_path = f"./checkpoints/best_model_trial_{best_trial_number}.pt"

    early_stopping = EarlyStopping(patience=1, verbose=False, delta=0.0, path=checkpoint_path)
    

    # trainer = HyperbandTrainer(model=model, labeled_trainloader=labeled_trainloader,
    #     unlabeled_trainloader=unlabeled_trainloader, val_loader=val_loader, test_loader=test_loader, 
    #     epochs=1, optimizer=None, early_stopping=early_stopping,
    #     lambda_u=study.best_params["lambda_u"], temperature=best_params["temperature"], threshold=study.best_params["threshold"],
    #     use_amp=best_params["use_amp"], device=device,
    #     trial=study.best_trial, final_eval=False)

    trainer = HyperbandTrainer(model=model, labeled_trainloader=labeled_trainloader,
        unlabeled_trainloader=None, val_loader=val_loader, test_loader=test_loader, 
        epochs=None, optimizer=None, early_stopping=early_stopping,
        lambda_u=None, threshold=None, temperature=None,
        use_amp=True, device=device,
        trial=study.best_trial)

    early_stopping.load_best_model(model, device)
    trainer._evaluate_and_log(labeled_trainloader, "Best Trial Training")
    trainer._evaluate_and_log(val_loader, "Best Trial Validation")
    trainer._evaluate_and_log(test_loader, "Best Trial Test")




if __name__ == "__main__":
    params = {"seed": 42,
              "epochs": 150,
              "label_ratio": 1.00,
              "batch_size": 256,
              "mu": 4,
              "patience": 30,
              "temperature": 1.0
              }
    
    set_seed(params["seed"])
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = build_dataloaders(label_ratio=params["label_ratio"], batch_size=params["batch_size"], mu=params["mu"], 
                                     data_seed=0)

    study = optuna.create_study(direction="maximize", 
                                pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=params["epochs"], reduction_factor=3),
                                sampler=optuna.samplers.TPESampler(seed=params["seed"]),
                                study_name="fixmatch_hyperband")

    study.optimize(lambda trial: objective(trial, data_loaders, params, device), n_trials=20)

    print("Best Trial")
    print(f"  number: {study.best_trial.number}")
    print(f"  value (best f1): {study.best_trial.value:.4f}")
    print(f"  params: {study.best_trial.params}")

    evaluate_best_trial(study, data_loaders, device=device)
