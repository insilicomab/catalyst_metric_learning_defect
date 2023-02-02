from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def get_classification_report(true, y_pred, int_to_label):
    true = [int_to_label[true_] for true_ in true]
    y_pred = [int_to_label[y_pred_] for y_pred_ in y_pred]
    return classification_report(true, y_pred)


def get_confusion_matrix(true, y_pred, labels):
    cm = confusion_matrix(true, y_pred, labels=labels)
    return cm