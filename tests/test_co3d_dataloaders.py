import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
from src.utils import co3d_dataloaders
import matplotlib.pyplot as plt

# image [batch, sequence, channel, height, width]
# depth [batch, sequence, height, width]
def save_and_visualize_batch(batch, output_dir="output_batch_visualization"):
    """Save all images and depth maps in the first sequence of the first batch with improved naming."""
    os.makedirs(output_dir, exist_ok=True)

    if "images" in batch and "depths" in batch and "seq_name" in batch and "ids" in batch:
        images = batch["images"]
        depths = batch["depths"]
        seq_name = batch["seq_name"]
        ids = batch["ids"]

        if isinstance(images, torch.Tensor) and images.ndimension() == 5 and isinstance(depths, torch.Tensor) and depths.ndimension() == 4:
            first_image_sequence = images[0]
            first_depth_sequence = depths[0]

            for i, (image, depth) in enumerate(zip(first_image_sequence, first_depth_sequence)):
                if image.size(0) in [3, 4]:
                    image_rgb = image.permute(1, 2, 0).numpy()  # Convert to HWC format
                    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())  # Normalize to [0, 1]
                    image_name = f"{seq_name[0]}_id{ids[0]}_frame{i}_rgb.png"
                    image_path = os.path.join(output_dir, image_name)
                    plt.imsave(image_path, image_rgb)
                    print(f"Saved RGB image: {image_path}")

                depth_map = depth.numpy()
                depth_name = f"{seq_name[0]}_id{ids[0]}_frame{i}_depth.png"
                depth_path = os.path.join(output_dir, depth_name)
                plt.imsave(depth_path, depth_map, cmap="viridis")
                print(f"Saved depth map: {depth_path}")
        else:
            print("Images or depths tensor is not in the expected format.")
    else:
        print("Batch does not contain required keys: 'images', 'depths', 'seq_name', 'ids'.")

def test_co3d_dataloaders():
    batch_size = 4
    num_workers = 2
    pin_memory = True
    DDP_mode = False

    dataloaders = co3d_dataloaders(batch_size, num_workers, pin_memory, DDP_mode)

    print("Testing train dataloader...")
    train_loader = dataloaders["train"]
    for i, batch in enumerate(train_loader):
        save_and_visualize_batch(batch)
        break

if __name__ == "__main__":
    test_co3d_dataloaders()
