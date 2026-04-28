import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.reproducibility import build_dataloader_generator, seed_worker

def get_dataloaders(data_dir, batch_size=32, seed=42):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )

    val = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transform
    )

    test = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=transform
    )

    generator = build_dataloader_generator(seed)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val, batch_size=batch_size, worker_init_fn=seed_worker)
    test_loader = DataLoader(test, batch_size=batch_size, worker_init_fn=seed_worker)

    return train_loader, val_loader, test_loader