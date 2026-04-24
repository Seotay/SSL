import torch
import logging

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
from dataset.dataset import get_wm811k
from model.model import ResNet18, ResNet10
from model.custom_model import CNN_Classifier, ResnetModel
from utils.utils import set_seed, EarlyStopping
from utils.trainer_v2 import Trainer
import numpy as np



logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


if __name__ == "__main__":
    set_seed(42)
    label_ratio = 1.00
    batch_size = 256
    mu = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_wm811k(
        labeled_path="./data/wm811k/preprocessing/labeled.pkl",
        unlabeled_path="./data/wm811k/preprocessing/unlabeled.pkl",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        image_size= 96,
        data_seed=0,
        label_ratio=label_ratio,

        cutout_num_holes=4,
        cutout_ratio=0.2,
        noise_prob=0.05,
    )
    

    logger.info(f"train_labeled_dataset: {len(train_labeled_dataset)}")
    logger.info(f"val_dataset: {len(val_dataset)}")
    logger.info(f"test_dataset: {len(test_dataset)}")
    logger.info(f"train_unlabeled_dataset: {len(train_unlabeled_dataset)}")
    
    
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=RandomSampler(train_labeled_dataset),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    unlabeled_trainloader = DataLoader(
        train_unlabeled_dataset,
        sampler=RandomSampler(train_unlabeled_dataset),
        batch_size=batch_size * mu,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        drop_last=False,
        )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),    
        batch_size=batch_size,
        drop_last=False
        )
    
    model = ResnetModel(num_classes=9).to(device)
    early_stopping = EarlyStopping(patience=150, verbose=True, delta=0.0, path="./checkpoints/best_model.pt")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    trainer = Trainer(
        model=model, 
        labeled_trainloader=labeled_trainloader, unlabeled_trainloader=unlabeled_trainloader,
        val_loader=val_loader, test_loader=test_loader,
        epochs=150,
        optimizer=optimizer, scheduler=scheduler, early_stopping=early_stopping,
        
        lambda_u=1.0, rampup_epochs=30, temperature=1.0, threshold=0.90,
        use_amp=True, device=device)

    trainer.training()