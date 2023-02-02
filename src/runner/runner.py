import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catalyst import dl, metrics
from tqdm import tqdm
from omegaconf import DictConfig

from .metrics import (
    get_classification_report, get_confusion_matrix
)


class CustomRunner(dl.Runner):
    def predict_batch(self, batch):
        # test step
        features, _ = batch
        return self.model.encoder(features.to(self.engine.device))


    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss", "accuracy", "recall", "precision", "f1_score", "fbeta_score"]
        }


    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        features, targets = batch
        # run model forward pass
        logits = self.model(features, targets)
        # compute the loss
        loss = self.criterion(logits, targets)
        # compute other metrics of interest
        accuracy = metrics.accuracy(logits, targets)
        recall = metrics.recall(logits, targets)
        precision = metrics.precision(logits, targets)
        f1_score = metrics.f1_score(logits, targets)
        fbeta_score = metrics.fbeta_score(logits, targets, beta=0.5)
        # log metrics
        self.batch_metrics.update(
            {"loss": loss, "accuracy": accuracy[0], "recall": recall.mean(), "precision": precision.mean(), "f1_score": f1_score.mean(), "fbeta_score": fbeta_score.mean()}
        )
        for key in ["loss", "accuracy", "recall", "precision", "f1_score", "fbeta_score"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )
        # run model backward pass
        if self.is_train_loader:
            self.engine.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()


    def on_loader_end(self, runner):
        for key in ["loss", "accuracy", "recall", "precision", "f1_score", "fbeta_score"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
    

    def test_step(self, loader, config: DictConfig):
        int2label = { v:k for k, v in config.label_map.items() }
        preds, targets = [], []
        for features, target in tqdm(loader):
            logits = self.model(
                features.to(self.engine.device),
                target.to(self.engine.device)
            )
            pred = logits.argmax(1).detach().cpu().numpy()
            preds.extend(pred)
            targets.extend(target.detach().cpu().numpy())

        self._evaluate(targets, preds, int2label)
    

    def _evaluate(self, true, y_pred, int_to_label):
        self._save_classification_report(true, y_pred, int_to_label)
        self._save_confusion_matrix(true, y_pred, int_to_label)


    def _save_classification_report(self, true, y_pred, int_to_label):
        """
        Save classification report to txt.
        """
        cls_report_str = get_classification_report(true, y_pred, int_to_label)
        with open('output/classification_report.txt', 'w') as f:
            f.write(cls_report_str)
    

    def _save_confusion_matrix(self, true, y_pred, int_to_label):
        """
        Save confusion matrix.
        """
        cm = get_confusion_matrix(true, y_pred, labels=np.arange(len(int_to_label)))
        df_cm = pd.DataFrame(cm, index=int_to_label.values(), columns=int_to_label.values())
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Prediction label')
        plt.ylabel('True label')
        plt.savefig('output/confusion_matrix.png')
