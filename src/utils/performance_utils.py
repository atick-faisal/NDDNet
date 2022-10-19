import numpy as np

from rich.table import Table
from rich.console import Console
from numpy.typing import NDArray

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PerformanceMetrics:
    def __init__(self):
        self.console = Console()
        self.table = Table(title="Performace Metrics")
        self.table.add_column("Subject ID")
        self.table.add_column("Precision")
        self.table.add_column("Recall")
        self.table.add_column("F1-Score")
        self.table.add_column("Accuracy")
        self.accuracies = []

    def add_entry(
        self, subject_id: str, 
        y_true: NDArray[np.uint8], 
        y_pred: NDArray[np.uint8]
        ):
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        self.accuracies.append(accuracy)
        self.table.add_row(
            subject_id,
            "{:.3f}".format(precision),
            "{:.3f}".format(recall),
            "{:.3f}".format(f1),
            "{:.3f}".format(accuracy),
        )

    def print_metrics(self):
        accuracy = np.mean(self.accuracies)
        self.table.add_row("", "", "", "", "{:.3f}".format(accuracy))
        self.console.print(self.table)
