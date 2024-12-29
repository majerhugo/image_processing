import torch
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_optimal_offsets(image_path, patch_size, stride):
    """
    Calculate offsets to distribute leftover pixels evenly when tiling an image.

    Args:
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        patch_size (int): Size of the square patches (in pixels).
        stride (int): Step size between patches (in pixels).

    Returns:
        (int, int): offset_left, offset_top
    """

    with rio.open(image_path) as img:
        img_width = img.width
        img_height = img.height

        if patch_size > img_width or patch_size > img_height:
            raise ValueError("Patch size cannot be larger than the image dimensions.")

        if stride <= 0:
            raise ValueError("Stride must be a positive integer.")

        leftover_width = img_width - ((img_width - patch_size) // stride * stride + patch_size)
        leftover_height = img_height - ((img_height - patch_size) // stride * stride + patch_size)

        offset_left = leftover_width // 2
        offset_top = leftover_height // 2

    return offset_left, offset_top

class ImagePatchesDataset(Dataset):
    def __init__(self, image_path, patch_size, stride, offset_left=0, offset_top=0):
        """
            Dataset for extracting patches from an image (features).

            Args:
                image_path (str): Path to the image (features).
                patch_size (int): Size of the square patches (in pixels), e.g. '32' will generate 32x32 patches.
                stride (int): Step size between patches (in pixels).
                offset_left (int | 'best'): Number of pixels to ignore from the left edge of the image.
                    If '0' - no offset will be used.
                    If 'best' - evenly distributed optimal offset will be calculated.
                offset_top (int | 'best'): Number of pixels to ignore from the top edge of the image.
                    If '0' - no offset will be used.
                    If 'best' - evenly distributed optimal offset will be calculated.
        """

        self.image_path = image_path
        self.patch_size = patch_size
        self.stride = stride

        if offset_left == 'best':
            self.offset_left, _ = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_left = offset_left

        if offset_top == 'best':
            _, self.offset_top = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_top = offset_top

        # open the imagery
        self.src_features = rio.open(image_path)

        # adjust dimensions based on offsets
        self.width = self.src_features.width - self.offset_left
        self.height = self.src_features.height - self.offset_top

        # Precompute patch positions (accounting for offsets)
        self.patches = [
            (row, col)
            for row in range(0, self.height - patch_size + 1, stride)
            for col in range(0, self.width - patch_size + 1, stride)
        ]

        print(f"Total patches generated: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]
        row += self.offset_top
        col += self.offset_left

        # Extract the image patch
        window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
        patch_features = self.src_features.read(window=window)  # Shape: (bands, patch_size, patch_size)

        return torch.tensor(patch_features, dtype=torch.float32)

class GroundTruthPatchesDataset(Dataset):
    def __init__(self, image_path, reference_path, patch_size, stride, offset_left=0, offset_top=0, background_label=0):
        """Dataset for extracting ground truth patches from a reference raster.

        Args:
            image_path (str): Path to the image (features).
            reference_path (str): Path to the reference raster (labels). Must be same resolution and extent as imagery.
            patch_size (int): Size of the square patches (in pixels), e.g. '32' will generate 32x32 patches.
            stride (int): Step size between patches (in pixels).
            offset_left (int | 'best'): Number of pixels to ignore from the left edge of the image.
                If '0' - no offset will be used.
                If 'best' - evenly distributed optimal offset will be calculated.
            offset_top (int | 'best'): Number of pixels to ignore from the top edge of the image.
                If '0' - no offset will be used.
                If 'best' - evenly distributed optimal offset will be calculated.
            background_label:

        """

        self.image_path = image_path
        self.reference_path = reference_path
        self.patch_size = patch_size
        self.stride = stride
        self.background_label = background_label

        if offset_left == 'best':
            self.offset_left, _ = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_left = offset_left

        if offset_top == 'best':
            _, self.offset_top = calculate_optimal_offsets(self.image_path, self.patch_size, self.stride)
        else:
            self.offset_top = offset_top

        # open the imagery and reference rasters
        self.src_features = rio.open(image_path)
        self.src_labels = rio.open(reference_path)

        # ensure the dimensions match
        if self.src_features.width != self.src_labels.width or self.src_features.height != self.src_labels.height:
            raise ValueError("Image and reference raster dimensions do not match!")

        # adjust dimensions based on offsets
        self.width = self.src_features.width - self.offset_left
        self.height = self.src_features.height - self.offset_top

        # Precompute patch positions (accounting for offsets)
        self.patches = []
        self.ground_truth_patches_count = 0  # Initialize count of ground truth patches

        for row in range(0, self.height - patch_size + 1, stride):
            for col in range(0, self.width - patch_size + 1, stride):

                # Extract the central pixel from the reference raster (label)
                center_row = row + self.offset_top + self.patch_size // 2
                center_col = col + self.offset_left + self.patch_size // 2
                central_window = rio.windows.Window(center_col, center_row, 1, 1)
                label = self.src_labels.read(1, window=central_window).item()

                # Only include patches with valid labels (non-background)
                if label != self.background_label:
                    self.patches.append((row, col))
                    self.ground_truth_patches_count += 1  # Increment count for valid patches

        print(f"Total ground truth patches generated: {self.ground_truth_patches_count}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]
        row += self.offset_top
        col += self.offset_left

        # Extract the image patch
        window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
        patch_features = self.src_features.read(window=window)  # Shape: (bands, patch_size, patch_size)

        # Extract the central pixel from the reference raster (label)
        center_row = row + self.patch_size // 2
        center_col = col + self.patch_size // 2
        central_window = rio.windows.Window(center_col, center_row, 1, 1)
        label = self.src_labels.read(1, window=central_window).item()  # Read as scalar

        # Return the patch and its label
        return torch.tensor(patch_features, dtype=torch.float32), label

def get_image_patches_loader(image_path, patch_size, stride, batch_size, offset_left=0, offset_top=0):
    dataset = ImagePatchesDataset(image_path, patch_size, stride, offset_left, offset_top)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_ground_truth_patches_loader(image_path, reference_path, patch_size, stride, batch_size, offset_left=0, offset_top=0, background_label=0):
    dataset = GroundTruthPatchesDataset(image_path, reference_path, patch_size, stride, offset_left, offset_top, background_label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Example usage:
image_path = r"E:\ml_dpz\composite_subset.tif"  # Path to Sentinel image (features)
reference_path = r"E:\ml_dpz\berlin_lcz_GT_fullres.tif"  # Path to reference raster (labels)
patch_size = 32
stride = 32
batch_size = 8
offset_left = 'best'
offset_top = 'best'
background_label=0

# Create data loaders
image_patches_loader = get_image_patches_loader(
    image_path=image_path,
    patch_size=patch_size,
    stride=stride,
    batch_size=batch_size,
    offset_left=offset_left,
    offset_top=offset_top
)

ground_truth_loader = get_ground_truth_patches_loader(
    image_path=image_path,
    reference_path=reference_path,
    patch_size=patch_size,
    stride=stride,
    batch_size=batch_size,
    offset_left=offset_left,
    offset_top=offset_top,
    background_label=background_label
)


for i, features in enumerate(image_patches_loader):
    print(f"Batch {i + 1} (All Patches): Features shape: {features.shape}")
    if i == 5:  # Limit to 3 batches for demonstration
        break

# Example: Iterate over ground truth patches
for i, (features, labels) in enumerate(ground_truth_loader):
    print(f"Batch {i + 1} (Ground Truth): Features shape: {features.shape}, Labels: {labels}")
    if i == 5:  # Limit to 3 batches for demonstration
        break