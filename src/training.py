from typing import Any
from typing import Iterable
from typing import Protocol
from collections.abc import Callable

from torch import Tensor 
from torch import argmax
from torch import device as Device
from torch import no_grad
from torch.nn import Module
from torch.optim import Optimizer 

class Metrics(Protocol):

    def to(self, device: Any) -> Metrics:...

    def reset(self) -> None:...

    def update(self, *args, **kwargs) -> None:...

    def compute(self) -> dict[str, Tensor]:... 

def fit(
    model: Module,
    criterion: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: Device,
    loader: Iterable[tuple[Tensor, Tensor]],
    metrics: Metrics,
) -> dict[str, Tensor]:
    model.train()
    metrics.reset()

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(features)            
        loss = criterion(output, targets)    
        loss.backward()
        optimizer.step()  
        metrics.update(loss, argmax(output, dim=1), targets)

    return metrics.compute()

def evaluate(
    model: Module,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    loader: Iterable[tuple[Tensor, Tensor]],
    metrics: Metrics,
)-> dict[str, Tensor]:
    model.eval()
    metrics.reset()

    with no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            output = model(features)           
            loss = criterion(output, targets)    
            metrics.update(loss, argmax(output, dim=1), targets) 
    return metrics.compute()