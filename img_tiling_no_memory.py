import os
import torch
import rasterio as rio
from torch.utils.data import Dataset

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

    # get image width and height
    with rio.open(image_path) as img:
        img_width = img.width
        img_height = img.height

        if patch_size > img_width or patch_size > img_height:
            raise ValueError("Patch size cannot be larger than the image dimensions.")

        if stride <= 0:
            raise ValueError("Stride must be a positive integer.")

        # compute leftover pixels according to patch size and stride
        leftover_width = img_width - ((img_width - patch_size) // stride * stride + patch_size)
        leftover_height = img_height - ((img_height - patch_size) // stride * stride + patch_size)

        # distribute leftover pixels equally
        offset_left = leftover_width // 2
        offset_top = leftover_height // 2

    return offset_left, offset_top

class ImagePatchesDataset(Dataset):
    def __init__(self, image_path, reference_path, patch_size, stride, offset_left=0, offset_top=0):
        """
        Dataset for extracting patches from an image and corresponding labels from a reference raster.

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
        """

        self.image_path = image_path
        self.reference_path = reference_path
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

        # open the imagery and reference rasters
        self.src_features = rio.open(image_path)
        self.src_labels = rio.open(reference_path)

        # ensure the dimensions match
        if self.src_features.width != self.src_labels.width or self.src_features.height != self.src_labels.height:
            raise ValueError("Image and reference raster dimensions do not match!")

        # adjust dimensions based on offsets
        self.width = self.src_features.width - self.offset_left
        self.height = self.src_features.height - self.offset_top

        #self.num_bands = self.src_features.count

        # Precompute patch positions (accounting for offsets)
        self.patches = [
            (row, col)
            for row in range(0, self.height - patch_size + 1, stride)
            for col in range(0, self.width - patch_size + 1, stride)
        ]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]

        # Adjust row and col to account for offsets
        row += self.offset_top
        col += self.offset_left

        # Extract the image patch
        window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
        patch_features = self.src_features.read(window=window)  # Shape: (bands, patch_size, patch_size)

        # Extract the central pixel from the reference raster
        center_row = row + self.patch_size // 2
        center_col = col + self.patch_size // 2
        central_window = rio.windows.Window(center_col, center_row, 1, 1)
        label = self.src_labels.read(1, window=central_window).item()  # Read as scalar

        # Return the patch and its label
        return torch.tensor(patch_features, dtype=torch.float32), label

def save_patches(patches_dataset, outdir_patches, outdir_gt, background_label=0):

    os.makedirs(outdir_patches, exist_ok=True)
    os.makedirs(outdir_gt, exist_ok=True)

    # paths to all Ground Truth patches
    gt_patches_paths = []

    # corresponding labels
    labels_list = []

    for idx, (features, label) in enumerate(patches_dataset):

        # Save every generated patch - later used for prediction
        patch_path = os.path.join(outdir_patches, f"patch_{str(idx).zfill(13)}.pt")
        torch.save(features, patch_path)

        # Save only Ground Truth patches (label != 0)
        if label != background_label:
            gt_patch_path = os.path.join(outdir_gt, f"patch_{str(idx).zfill(13)}.pt")
            torch.save(features, gt_patch_path)

            gt_patches_paths.append(gt_patch_path)
            labels_list.append(label)

        # Log progress
        if idx % 10000 == 0:
            print(f"Processed tile {idx}")

    # Save labels for valid tiles
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    torch.save(labels_tensor, os.path.join(outdir_gt, "labels.pt"))

    print(f"Saved {len(gt_patches_paths)} Ground Truth patches to '{outdir_gt}'.")
    print(f"Saved all patches to '{outdir_patches}'.")

def GenerateImagePatches(image_path, reference_path, patch_size, stride, outdir_patches, outdir_gt,
                         offset_left=0, offset_top=0, background_label=0):

    # Initialize the dataset
    dataset = ImagePatchesDataset(
        image_path=image_path,
        reference_path=reference_path,
        patch_size=patch_size,
        stride=stride,
        offset_left=offset_left,
        offset_top=offset_top
    )

    return save_patches(
        patches_dataset=dataset,
        outdir_patches=outdir_patches,
        outdir_gt=outdir_gt,
        background_label=background_label
    )


def calculate_patches(image_path, patch_size, stride, offset_left, offset_top):
    """
    Calculate the number of patches in a row and column for an image.

    Args:
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        patch_size (int): Size of the square patches (in pixels).
        stride (int): Step size between patches (in pixels).
        offset_left (int): Offset from the left of the image in pixels.
        offset_top (int): Offset from the top of the image in pixels.

    Returns:
        (int, int): Number of patches in a row, number of patches in a column
    """

    # get image width and height
    img = rio.open(image_path)
    img_width = img.width
    img_height = img.height

    # pozor na +1!
    num_patches_row = ((img_width - offset_left - patch_size) // stride) + 1
    num_patches_col = ((img_height - offset_top - patch_size) // stride) + 1

    return num_patches_row, num_patches_col