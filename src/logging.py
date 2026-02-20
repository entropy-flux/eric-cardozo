from torch.nn import Module  
from torchsystem.registry import getname, getarguments
 
from pathlib import Path
from csv import DictWriter 
 
class Logger:
    def __init__(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(path, "w", newline="")
        self.writer = DictWriter(
            self.file,
            fieldnames=["epoch", "phase", "loss", "accuracy", "lr"]
        )
        self.writer.writeheader()

    def log(self, epoch: int, phase: str, metrics: dict, lr: float):
        self.writer.writerow({
            "epoch": epoch,
            "phase": phase,
            "loss": metrics["loss"].item(),
            "accuracy": metrics["accuracy"].item(),
            "lr": lr,
        })

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def nameof(model: Module, seed: int) -> str:
    key_aliases = {
        "image_size": "imgz",
        "patch_size": "pz",
        "model_dimension": "dim",
        "hidden_dimension": "hdim",
        "number_of_heads": "nhead",
        "number_of_layers": "nlayer",
        "number_of_classes": "ncls",
        "dropout_rate": "p",
        "attention_dropout_rate": "ap",
    }

    name = f"{getname(model)}-s{seed}"
    parts = [name]
    for k, v in sorted(getarguments(model).items()):
        k = key_aliases.get(k, k)
        v = "x".join(map(str, v)) if isinstance(v, tuple) else v
        parts.append(f"{k}={v}")
    return "-".join(parts)