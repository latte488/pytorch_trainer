from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def mnist(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader
