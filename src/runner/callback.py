from catalyst import dl
from omegaconf import DictConfig

def get_callbacks(config: DictConfig) -> list:
    callback_list = []

    if config.callbacks.early_stopping.enable:
        earlystopping = dl.EarlyStoppingCallback(
            patience=config.callbacks.early_stopping.patience, 
            loader_key=config.callbacks.early_stopping.loader_key,
            metric_key=config.callbacks.early_stopping.metric_key,
            minimize=config.callbacks.early_stopping.minimize,
            min_delta=config.callbacks.early_stopping.min_delta
        )
        callback_list.append(earlystopping)

    if config.callbacks.model_checkpoint.enable:
        checkpoint = dl.CheckpointCallback(
            logdir=config.callbacks.model_checkpoint.logdir,
            loader_key=config.callbacks.model_checkpoint.loader_key,
            metric_key=config.callbacks.model_checkpoint.metric_key,
            minimize=config.callbacks.model_checkpoint.minimize,
            topk=config.callbacks.model_checkpoint.topk,
        )

    callback_list.append(checkpoint)

    return callback_list

