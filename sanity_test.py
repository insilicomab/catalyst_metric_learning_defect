from catalyst.utils import set_global_seed, prepare_cudnn
import hydra
from omegaconf import DictConfig

from src.dataset import get_loaders
from src.model import EncoderWithHead, get_loss_fn, get_optimizer, get_scheduler
from src.runner import CustomRunner, get_callbacks
from src.utils import data_split


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
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
    optimizer = get_optimizer(config=cfg, net=model)
    scheduler=get_scheduler(config=cfg, optimizer=optimizer)
    loss_fn = get_loss_fn(config=cfg)

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
        verbose=True,
        load_best_on_end=True,
    )

    # test step
    runner.test_step(loader=loaders['valid'], config=cfg)


if __name__ == "__main__":
    main()