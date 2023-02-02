import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import OrderedDict

from .dataset import ImageDataset
from .transformation import Transforms


def get_loaders(
    config: DictConfig,
    x_train: list,
    y_train: list,
    x_val: list,
    y_val: list
    ) -> OrderedDict[str, DataLoader]:
    train_dataset = ImageDataset(
        image_name_list=x_train,
        label_list=y_train,
        img_dir=config.img_dir,
        transform=Transforms(config=config),
        phase='train',
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_dataloader.batch_size,
        shuffle=config.train_dataloader.shuffle,
        num_workers=config.train_dataloader.num_workers,
        pin_memory=config.train_dataloader.pin_memory,
    )

    val_dataset = ImageDataset(
        image_name_list=x_val,
        label_list=y_val,
        img_dir=config.img_dir,
        transform=Transforms(config=config),
        phase='val',
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val_dataloader.batch_size,
        shuffle=config.val_dataloader.shuffle,
        num_workers=config.val_dataloader.num_workers,
        pin_memory=config.val_dataloader.pin_memory,
    )

    return {"train": train_dataloader, "valid": val_dataloader}