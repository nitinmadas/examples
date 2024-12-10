import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="gloo", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.epoch_start = 0


    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for source, targets in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                _, predicted = torch.max(output, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        self.model.train()
        return correct / total    

    def train(self, max_epochs: int):
        for epoch in range(self.epoch_start, max_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            for batch_idx, (source, targets) in enumerate(self.train_data):
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                loss = self._run_batch(source, targets)
                running_loss += loss
            
            train_loss = running_loss / len(self.train_data)
            val_accuracy = self._evaluate()

            print(f" GPU{self.gpu_id} Epoch {epoch} | Loss: {train_loss:.4f} | Accuracy: {val_accuracy:.4f}")


def prepare_dataloader(batch_size: int, sample_size: int = 1000):
    """
    Prepares the train and validation DataLoader with a subset of CIFAR-10 for faster testing.

    Args:
        batch_size (int): Number of samples per batch.
        sample_size (int): Number of samples to use from the dataset.

    Returns:
        tuple: Train and validation DataLoader.
    """
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # Load full datasets
    full_train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    full_val_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Take a subset of the train and validation sets
    train_set = torch.utils.data.Subset(full_train_set, list(range(sample_size)))
    val_set = torch.utils.data.Subset(full_val_set, list(range(sample_size // 10)))  # Smaller validation set

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_set),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(val_set),
        pin_memory=True
    )
    return train_loader, val_loader


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_loader, val_loader = prepare_dataloader(batch_size)
    model = torchvision.models.resnet18(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    trainer = Trainer(model, train_loader, val_loader, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    total_epochs = 5
    batch_size = 128
    save_every = 1
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)
