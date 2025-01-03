import torch
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
from rasterio.io import MemoryFile
import rioxarray
import numpy as np
import matplotlib.pyplot as plt

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

def match_rasters(raster_to_change_path, raster_path):
    raster_to_change = rioxarray.open_rasterio(raster_to_change_path, masked=True)
    raster = rioxarray.open_rasterio(raster_path, masked=True)
    raster_to_change = raster_to_change.drop_vars("band").squeeze()
    raster = raster.drop_vars("band").squeeze()

    def print_raster(raster):
        print(
            f"shape: {raster.rio.shape}\n"
            f"resolution: {raster.rio.resolution()}\n"
            f"bounds: {raster.rio.bounds()}\n"
            f"sum: {raster.sum().item()}\n"
            f"CRS: {raster.rio.crs}\n"
        )

    print("Matching this Raster:\n----------------\n")
    print_raster(raster_to_change)
    print("To this Raster:\n----------------\n")
    print_raster(raster)

    raster_matched = raster_to_change.rio.reproject_match(raster)

    print("Matched Raster:\n-------------------\n")
    print_raster(raster_matched)
    print("To this Raster:\n----------------\n")
    print_raster(raster)

    # for debug use
    #return raster_matched.rio.to_raster("debug.tif", driver="GTiff", compress="LZW")

    # Write the aligned raster to a rasterio.MemoryFile
    with MemoryFile() as memfile:
        with memfile.open(
                driver="GTiff",
                height=raster_matched.rio.shape[0],
                width=raster_matched.rio.shape[1],
                count=1,
                dtype=raster_matched.dtype,
                crs=raster_matched.rio.crs,
                transform=raster_matched.rio.transform(),
        ) as dataset:
            dataset.write(raster_matched.values, 1)  # Write the data to the in-memory dataset

        return memfile.open()

class ImagePatchesDataset(Dataset):
    def __init__(self, image_path, patch_size, stride, offset_left=0, offset_top=0, reference_path=None, background_label=None):
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
        with rio.open(image_path) as src_features:
            self.src_features = src_features

            # adjust dimensions based on offsets
            self.width = src_features.width - self.offset_left
            self.height = src_features.height - self.offset_top

        # reference raster and background label provided
        if reference_path and background_label is not None:

            print('Reference labels were provided, generating ground truth patches...')

            # open reference raster
            self.src_labels = rio.open(reference_path)  # Open the reference raster here

            self.background_label = background_label

            # ensure the dimensions and CRS match
            if self.src_features.width != self.src_labels.width \
                    or self.src_features.height != self.src_labels.height \
                    or self.src_features.crs != self.src_labels.crs:
                print("Dimensions or CRS do not match, aligning the reference raster to match the features raster...")

                # Use match_rasters to align the reference raster
                self.src_labels = match_rasters(reference_path, self.image_path)

            # precompute patch positions (accounting for offsets)
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

        # reference raster and background label not provided - just tiling the image
        else:

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
        with rio.open(self.image_path) as src_features:
            window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch_features = src_features.read(window=window)  # Shape: (bands, patch_size, patch_size)

        # check if the reference raster was loaded
        if hasattr(self, 'src_labels'):
            # Extract the central pixel from the reference raster (label)
            center_row = row + self.patch_size // 2
            center_col = col + self.patch_size // 2
            central_window = rio.windows.Window(center_col, center_row, 1, 1)
            label = self.src_labels.read(1, window=central_window).item()

            # Return the patch and its label
            return torch.tensor(patch_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # If no reference raster, return only the image patch
        return torch.tensor(patch_features, dtype=torch.float32)

def GenerateImagePatchesLoaders(image_path, patch_size, stride, batch_size, offset_left=0, offset_top=0, reference_path=None, background_label=None):
    """
    Generate DataLoaders for image patches and optionally for ground truth labels.

    Args:
        image_path (str): Path to the image (features).
        patch_size (int): Size of the square patches (in pixels).
        stride (int): Step size between patches (in pixels).
        batch_size (int): Number of patches per batch.
        offset_left (int | 'best'): Offset from the left edge of the image.
        offset_top (int | 'best'): Offset from the top edge of the image.
        reference_path (str, optional): Path to the reference raster (labels). Default is None.
        background_label (int, optional): Label value to ignore as background. Default is None.

    Returns:
        If reference_path and background_label are provided:
            (DataLoader, DataLoader): Features DataLoader and Ground Truth DataLoader.
        Otherwise:
            DataLoader: Features DataLoader only.
    """

    # Create the features dataset
    features_dataset = ImagePatchesDataset(image_path=image_path,
                                           patch_size=patch_size,
                                           stride=stride,
                                           offset_left=offset_left,
                                           offset_top=offset_top
                                           )

    # If reference_path and background_label are provided, create a ground truth dataset
    if reference_path and background_label is not None:

        gt_dataset = ImagePatchesDataset(image_path=image_path,
                                         patch_size=patch_size,
                                         stride=stride,
                                         offset_left=offset_left,
                                         offset_top=offset_top,
                                         reference_path=reference_path,
                                         background_label=background_label
                                         )

        # Return both datasets
        return DataLoader(features_dataset, batch_size=batch_size, shuffle=False), DataLoader(gt_dataset, batch_size=batch_size, shuffle=False)

    # If reference_path or background_label is not provided, return only the features dataset
    return DataLoader(features_dataset, batch_size=batch_size, shuffle=False)

def remap_labels(gt_loader):
    """
    Remap labels in the ground truth loader to a continuous range starting from 0 (PyTorch requirement)
    This function will overwrite the labels while iterating over the DataLoader.

    Args:
        gt_loader (DataLoader): Ground truth DataLoader with the original labels.

    Returns:
        DataLoader: Ground Truth DataLoader with remapped labels.
        label_mapping (dict): Remapping map.
    """

    # Print unique original labels
    all_labels = []
    for _, label in gt_loader:
        all_labels.extend(label.numpy())

    original_labels, counts = np.unique(all_labels, return_counts=True)
    print("Original unique label values: ", original_labels, counts)

    # Create a mapping from original labels to a continuous range starting from 0
    label_mapping = {label: idx for idx, label in enumerate(original_labels)}

    # Create new Torch Dataset with remapped labels
    class RemappedDataset(Dataset):
        def __init__(self, original_loader, label_mapping):
            self.original_loader = original_loader
            self.label_mapping = label_mapping

        def __len__(self):
            return len(self.original_loader.dataset)

        def __getitem__(self, idx):
            features, label = self.original_loader.dataset[idx]

            remapped_label = torch.tensor(self.label_mapping.get(label.item(), -1))

            return features, remapped_label

    # Create the remapped dataset
    remapped_dataset = RemappedDataset(gt_loader, label_mapping)

    # Create a new DataLoader with the remapped labels
    gt_loader_remapped = DataLoader(remapped_dataset, batch_size=gt_loader.batch_size, shuffle=False)

    # Print remapped unique labels
    all_labels = []
    for _, label in gt_loader_remapped:
        all_labels.extend(label.numpy())

    remapped_labels, counts = np.unique(all_labels, return_counts=True)
    print("Remapped unique label values: ", remapped_labels, counts)

    return gt_loader_remapped, label_mapping