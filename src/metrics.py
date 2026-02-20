from torch import Tensor, device as Device 
from torcheval.metrics import MulticlassAccuracy as Accuracy, Mean
 
class Metrics:
    def __init__(self):
        self.accuracy = Accuracy(num_classes=10)
        self.loss = Mean()

    def to(self, device: Device) -> Metrics:
        self.accuracy.to(device)
        self.loss.to(device)
        return self

    def update(self, loss: Tensor, predictions: Tensor, targets: Tensor) -> None:
        self.accuracy.update(predictions, targets)
        self.loss.update(loss)

    def reset(self) -> None:
        self.accuracy.reset()
        self.loss.reset()

    def compute(self) -> dict[str, Tensor]:
        return {
            "loss": self.loss.compute(),
            "accuracy": self.accuracy.compute(),
        } 