from catalyst import dl


class CustomWandbLogger(dl.WandbLogger):
    def close_log(self, scope: str = None) -> None:
        pass