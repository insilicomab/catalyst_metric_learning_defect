import os
import shutil
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst import dl
import hydra
from omegaconf import DictConfig
import wandb

from src.dataset import get_loaders
from src.model import EncoderWithHead, get_loss_fn, get_optimizer, get_scheduler
from src.runner import CustomRunner, CustomWandbLogger, get_callbacks
from src.utils import data_split


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    # initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config={
            'data': os.path.basename(cfg.wandb.data_dir),
            'model': cfg.wandb.model_name,
            'layer': cfg.wandb.layer_name,
        }
    )

    # save crucial artifacts before training begins
    shutil.copy2('config/config.yaml', os.path.join(wandb.run.dir, 'hydra_config.yaml'))
    
    # data split
    x_train, x_val, y_train, y_val = data_split(config=cfg)

    # set seed
    set_global_seed(seed=cfg.seed)
    prepare_cudnn(deterministic=True)

    # loaders
    loaders = get_loaders(
        config=cfg,
        x_train=x_train, 
        y_train=y_train,
        x_val=x_val, 
        y_val=y_val,
    )

    # model
    model = EncoderWithHead(
        model_name=cfg.encoder.model_name,
        pretrained=cfg.encoder.pretrained,
        layer_name=cfg.layer.name,
        embedding_size=cfg.embedding_size,
        num_classes=cfg.num_classes,
    )

    # loss function & optimizer & scheduler
    loss_fn = get_loss_fn(config=cfg)
    optimizer = get_optimizer(config=cfg, net=model)
    scheduler = get_scheduler(config=cfg, optimizer=optimizer)

    # runner
    runner = CustomRunner()

    # train
    runner.train(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        seed=cfg.seed,
        num_epochs=cfg.runner.num_epochs,
        callbacks=get_callbacks(config=cfg),
        loggers={"wandb": CustomWandbLogger(project=cfg.wandb.project)},
        verbose=True,
        load_best_on_end=True,
    )

    shutil.copy2(
        'model/pth/model.best.pth',
        os.path.join(wandb.run.dir,'model_best.pth')
    )

    # test step
    runner.test_step(loader=loaders['valid'], config=cfg)
    shutil.copy2(
        'output/classification_report.txt',
        os.path.join(wandb.run.dir,'classification_report.txt')
    )
    shutil.copy2(
        'output/confusion_matrix.png',
        os.path.join(wandb.run.dir,'confusion_matrix.png')
    )


if __name__ == "__main__":
    main()