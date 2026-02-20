from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomErasing,
    ColorJitter
)

class CIFAR(Dataset):
    def __init__(self, train: bool) -> None:
        super().__init__()

        self.transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            ),
            RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            )
        ]) if train else Compose([
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        self.dataset = CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=self.transform
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return image, label

    def __len__(self) -> int:
        return len(self.dataset) 