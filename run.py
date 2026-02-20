import random
import torch

from src.models import ViT
from src.models.SwissGLUViT import SwissGLUViT
from src.models.CanonicalViT import CanonicalViT
from src.models.GrandCanonicalViT import GrandCanonicalViT
from src.models.TGCViT import GrandCanonicalTermalViT
from src.datasets.cifar import CIFAR
from src.training import fit, evaluate

from torch import save, device as Device
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader 
from torchsystem.registry import register  

from src.logging import Logger, nameof
from src.metrics import Metrics

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  

register(ViT)
register(SwissGLUViT)     
register(CanonicalViT)  
register(GrandCanonicalViT) 
register(GrandCanonicalTermalViT)
EPOCHS = 300

def run(model: Module, loaders: dict, device: Device):
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=300,
        eta_min=1e-5
    )

    metrics = Metrics().to(device)
    name = nameof(model, seed)
    logger = Logger(f"logs/{name}.csv")
    print("Model:", name)
    try:
        for epoch in range(EPOCHS):
            print("Epoch: ", {epoch})
            results = fit(
                model, criterion, optimizer,
                device, loaders["train"], metrics
            ) 

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            logger.log(epoch, "train", results, lr)

            results = evaluate(
                model, criterion,
                device, loaders["evaluation"], metrics
            )
            logger.log(epoch, "evaluation", results, lr) 
            logger.flush()

    finally:
        save(model.state_dict(), f"weights/{name}.pth")
        logger.close()  
  
if __name__ == "__main__": 
    device = Device("cuda")

    loaders = {
        "train": DataLoader(CIFAR(train=True), batch_size=128, shuffle=True),
        "evaluation": DataLoader(CIFAR(train=False), batch_size=128, shuffle=False)
    }   
    
    model = GrandCanonicalTermalViT(
        image_size=(32, 32),
        patch_size=(4, 4),
        model_dimension=64,
        hidden_dimension=128,
        number_of_heads=4,
        number_of_layers=4,
        number_of_classes=10,
        number_of_channels=3,
        dropout_rate=0.3
    ).to(device) 
    run(model, loaders, device)     

    model = GrandCanonicalTermalViT(
        image_size=(32, 32),
        patch_size=(4, 4),
        model_dimension=64,
        hidden_dimension=256,
        number_of_heads=4,
        number_of_layers=4,
        number_of_classes=10,
        number_of_channels=3,
        dropout_rate=0.3,
    ).to(device) 
    run(model, loaders, device)   

    model = GrandCanonicalTermalViT(
        image_size=(32, 32),
        patch_size=(4, 4),
        model_dimension=128,
        hidden_dimension=256,
        number_of_heads=4,
        number_of_layers=4,
        number_of_classes=10,
        number_of_channels=3,
        dropout_rate=0.4
    ).to(device)  
    run(model, loaders, device)  