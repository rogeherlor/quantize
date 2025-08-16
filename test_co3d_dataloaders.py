import torch
from src.utils import co3d_dataloaders

def test_co3d_dataloaders():
    batch_size = 4
    num_workers = 2
    pin_memory = True
    DDP_mode = False

    # Initialize dataloaders
    dataloaders = co3d_dataloaders(batch_size, num_workers, pin_memory, DDP_mode)

    # Test train dataloader
    print("Testing train dataloader...")
    train_loader = dataloaders["train"]
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}: {batch}")
        if i == 2:  # Test only the first 3 batches
            break

    # Test validation dataloader
    print("Testing validation dataloader...")
    val_loader = dataloaders["val"]
    for i, batch in enumerate(val_loader):
        print(f"Batch {i}: {batch}")
        if i == 2:  # Test only the first 3 batches
            break

if __name__ == "__main__":
    test_co3d_dataloaders()
