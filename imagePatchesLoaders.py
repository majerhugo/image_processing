# todo: fcia, kt. bude normalizovat patchky v DataLoaderoch
    # mean a std na normalizaciu treba vzdy odvodit len z trenovacich dat a nasledne pomocou nich normnalizovat
    # aj testovacie data a aj tie, na ktorych bude pustena predikcia
    # ASI HOTOVO
# todo: fcia, kt. bude augmentovat treningove patchky v DataLoaderu
    # CHECKNUT CI JE SPRAVNE SPRAVENA...
# todo: funkcia na remap labels naspat (vstup bude vektor predikcie)
# todo: zmenit calculate_optimal_offsets aby nemusela otvarat image z filepathu, ale aby mala na vstupe uz otvoreny
#  image alebo rovno dimenzie obrazku, nasledne bude treba upravit FeaturePatchesDataset a LabeledPatchesDataset aby
#  sa tato funkcia volala az po otvoreni obrazku
# todo: fcia, ktora vezme vektorove GT polygony a spravi stratif. rozdelenie na train/test (tuto fciu mozno netreba)
# todo: fcia na rasterizaciu GT polygonov, ktore boli apriori rozdelene na train/test (tato bude treba tak ci tak)
# todo: fcia, kt. po klasifikacii patchiek ich posklada naspat do mapy (tam potencialne vyuzit fciu calculate_patches)
# todo: fcia, kt. bude generovat patchky ne do DataLoaderov, ale ulozi ich na disk ako .pt alebo .npy, pripadne .h5,
#  tu sa bude dat zrecyklovat fcia z img_tiling_no_memory.py
# todo: fcia, kt. bude generovat patchky na disk ako GeoTiffy (zrecyklovat zase stary kod)

# todo: implementovat fcie pre generaciu patchiek na semanticku segmentaciu

import torch
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
from rasterio.io import MemoryFile
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.transforms import v2 as transforms
torch.set_printoptions(precision=6, sci_mode=False)

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
    print(f"Original unique label values:  {original_labels}, Counts: {counts}")

    # Create a mapping from original labels to a continuous range starting from 0
    label_mapping = {label: new_label for new_label, label in enumerate(original_labels)}

    # Create new Torch Dataset with remapped labels
    class RemappedDataset(Dataset):
        def __init__(self, original_loader, label_mapping):
            self.original_loader = original_loader
            self.label_mapping = label_mapping

        def __len__(self):
            return len(self.original_loader.dataset)

        def __getitem__(self, idx):
            features, label = self.original_loader.dataset[idx]

            # from created label map get corresponding new label
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
    print(f"Remapped unique label values: {remapped_labels}, Counts: {counts}")

    return gt_loader_remapped, label_mapping

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def visualize_labeled_patches(gt_loader, patches_per_class=2, bands=None):
    """
    Visualize defined number of labeled patches for each class from the DataLoader of labeled classes.

    Args:
        gt_loader (DataLoader): DataLoader of labeled patches.
        patches_per_class (int): Number of patches to display per class. Patches are selected randomly. (default=2)
        bands (tuple): Band order for composite visualization, eg. (0,1,2) will be RGB composite. If not specified
            greyscale patches will be displayed (default).

    """

    # Dictionary to store patches by label
    class_patches = {}

    # Iterate through the DataLoader and collect patches for each class
    for features, labels in gt_loader:

        # get the size of current batch
        batch_size = features.size(0)

        # Iterate through all patches in the batch
        for patch_index in range(batch_size):

            # Get label for the current patch
            label = labels[patch_index].item()

            # if label is not in dictionary, initialize an empty list for patches
            if label not in class_patches:
                class_patches[label] = []

            # append the patch to the corresponding class list
            class_patches[label].append(features[patch_index])

    # set up the plotting grid: columns = num_classes; rows = patches per class
    num_classes = len(class_patches)

    # sort dictionary
    class_patches = {key: value for key, value in sorted(class_patches.items())}

    fig, axes = plt.subplots(nrows=patches_per_class, ncols=num_classes)

    # ensure axes is iterable even when there's only one class
    if num_classes == 1:
        axes = [axes]

    # Loop through each class and select random patches to display
    for idx, (label, patches) in enumerate(class_patches.items()):
        # Select random patches for the current class
        selected_patches = random.sample(patches, patches_per_class)

        # Loop through selected patches and plot them
        for j, patch in enumerate(selected_patches):
            ax = axes[j][idx] if num_classes > 1 else axes[idx]

            # color composite patches
            if bands is not None:
                b1 = normalize(patch[bands[0]].squeeze().numpy())
                b2 = normalize(patch[bands[1]].squeeze().numpy())
                b3 = normalize(patch[bands[2]].squeeze().numpy())

                patch_image = np.array([b1, b2, b3])
                patch_image = np.transpose(patch_image, axes=(1, 2, 0))

                # highlight center pixel
                center_row = patch_image.shape[0] // 2
                center_col = patch_image.shape[1] // 2
                patch_image[center_row, center_col, :] = 1

                ax.imshow(patch_image)
                ax.set_title(f"Class {label} - Patch {j + 1}")
                ax.axis('off')

            # grayscale patches
            else:
                # patch to numpy array and remove the channel dimensions
                patch_image = normalize(patch[0].squeeze().numpy())

                # highlight center pixel
                center_row = patch_image.shape[0] // 2
                center_col = patch_image.shape[1] // 2
                patch_image[center_row, center_col] = 1

                ax.imshow(patch_image, cmap='gray')
                ax.set_title(f"Class {label} - Patch {j + 1}")
                ax.axis('off')

    # maximizing figure size
    try:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    except:
        plot_backend = plt.get_backend()
        if plot_backend == 'TkAgg':
            mng.resize(*mng.window.maxsize())
        elif plot_backend == 'wxAgg':
            mng.frame.Maximize(True)
        elif plot_backend == 'Qt4Agg':
            mng.window.showMaximized()

    plt.tight_layout()
    plt.show()

def get_normalization_parameters(training_loader):
    mean = 0
    std = 0
    total_pixels = 0

    # get mean and std for each batch in loader
    for patches, _ in training_loader:

        # flatten the patches tensor to (batch_size, C, H*W)
        pixels = patches.view(patches.size(0), patches.size(1), -1)

        mean += pixels.mean(dim=(0, 2)) * patches.size(0)
        std += pixels.std(dim=(0, 2)) * patches.size(0)

        total_pixels += patches.size(0)

    # get the resulting mean and std
    mean /= total_pixels
    std /= total_pixels

    return mean, std

def normalize_loader(original_loader, means, stds):

    for batch in original_loader:

        # loader with labels
        if len(batch) == 2:
            # create new Torch Dataset with labels
            class NormalizedDataset(Dataset):
                def __init__(self, original_loader, means, stds):
                    self.original_loader = original_loader
                    self.means = means
                    self.stds = stds
                    self.normalization = transforms.Normalize(mean=self.means, std=self.stds)

                def __len__(self):
                    return len(self.original_loader.dataset)

                def __getitem__(self, idx):
                    features, label = self.original_loader.dataset[idx]

                    norm_features = self.normalization(features)

                    return norm_features, label

        # loader without labels
        else:
            # create new Torch Dataset without labels
            class NormalizedDataset(Dataset):
                def __init__(self, original_loader, means, stds):
                    self.original_loader = original_loader
                    self.means = means
                    self.stds = stds
                    self.normalization = transforms.Normalize(mean=self.means, std=self.stds)

                def __len__(self):
                    return len(self.original_loader.dataset)

                def __getitem__(self, idx):
                    features = self.original_loader.dataset[idx]

                    norm_features = self.normalization(features)

                    return norm_features

        norm_dataset = NormalizedDataset(original_loader, means, stds)
        norm_loader = DataLoader(norm_dataset, batch_size=original_loader.batch_size, shuffle=False)

        return norm_loader

# checknut ci je spravne spravena...
def augment_loader(original_loader, transform, num_augmented_samples):
    """
    Augments the features in the dataloader and includes the original samples.

    Parameters:
    - original_loader (DataLoader): The original dataloader with features and labels.
    - transform (transform): PyTorch transformation object that defines augmentation (e.g., flipping, rotation).
    - num_augmented_samples (int): Number of augmented versions to create for each feature, including the original.
    """

    class AugmentedDataset(Dataset):
        def __init__(self, original_loader, transform, num_augmented_samples):
            self.original_loader = original_loader
            self.transform = transform
            self.num_augmented_samples = num_augmented_samples

        def __len__(self):
            return len(self.original_loader.dataset) * (self.num_augmented_samples + 1)  # +1 to include the original sample

        def __getitem__(self, idx):
            # Determine if this is an original or augmented sample
            original_idx = idx // (self.num_augmented_samples + 1)  # original sample index
            augmented_idx = idx % (self.num_augmented_samples + 1)  # augmented sample index (or original)

            features, label = self.original_loader.dataset[original_idx]

            if augmented_idx == 0:
                # Return the original sample
                return features, label
            else:
                # Apply the transformation to generate an augmented sample
                augmented_features = self.transform(features)
                return augmented_features, label

    # Create a new augmented dataset (including the original samples)
    augmented_dataset = AugmentedDataset(original_loader, transform, num_augmented_samples)
    print(f"Augmented dataset has {len(augmented_dataset)} patches.")

    # Create a new DataLoader with the augmented dataset
    augmented_loader = DataLoader(augmented_dataset, batch_size=original_loader.batch_size, shuffle=True)

    return augmented_loader

class FeaturePatchesDataset(Dataset):
    def __init__(self, image_path, patch_size, stride, offset_left=0, offset_top=0):
        """
            Dataset for extracted featured patches from an image.

            Args:
                image_path (str): File path to the feature image.
                patch_size (int): Size of the square patches (in pixels), e.g. '32' will generate 32x32 patches.
                stride (int): Step size between patches (in pixels).
                offset_left (int | 'best'): Number of pixels to ignore from the left edge of the image.
                    If '0' - no offset will be used (default).
                    If 'best' - evenly distributed optimal offset from both sides of the image will be calculated.
                offset_top (int | 'best'): Number of pixels to ignore from the top edge of the image.
                    If '0' - no offset will be used (default).
                    If 'best' - evenly distributed optimal offset from top and bottom of the image will be calculated.
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

        # precompute patch positions (accounting for offsets)
        self.patches = [
            (row + self.offset_top, col + self.offset_left)
            for row in range(0, self.height - patch_size + 1, stride)
            for col in range(0, self.width - patch_size + 1, stride)
        ]

        print(f"Total patches loaded: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]

        # extract the image patch features
        with rio.open(self.image_path) as src_features:
            window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch_features = src_features.read(window=window)

        return torch.tensor(patch_features, dtype=torch.float32)

class LabeledPatchesDataset(Dataset):
    def __init__(self, image_path, reference_path, patch_size, stride, offset_left=0, offset_top=0, background_label=0):
        """
            Dataset for extracting labeled patches from an image. Patch label corresponds to the label \
            of the central pixel of the patch derived from reference raster. If the central pixel label corresponds to
            provided background label, this patch is skipped.

            Args:
                image_path (str): Path to the feature image.
                reference_path (str): Path to the reference raster.
                patch_size (int): Size of the square patches (in pixels), e.g. '32' will generate 32x32 patches.
                stride (int): Step size between patches (in pixels).
                offset_left (int | 'best'): Number of pixels to ignore from the left edge of the image.
                    If '0' - no offset will be used (default).
                    If 'best' - evenly distributed optimal offset from both sides of the image will be calculated.
                offset_top (int | 'best'): Number of pixels to ignore from the top edge of the image.
                    If '0' - no offset will be used.
                    If 'best' - evenly distributed optimal offset from top and bottom of the image will be calculated.
                background_label: label corresponding to the background (nodata) of the reference raster (default=0).
        """

        self.image_path = image_path
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

        # open reference raster
        self.src_labels = rio.open(reference_path)

        # open the imagery
        with rio.open(image_path) as src_features:
            self.src_features = src_features

            # ensure the dimensions and CRS match
            if self.src_features.width != self.src_labels.width \
                    or self.src_features.height != self.src_labels.height \
                    or self.src_features.crs != self.src_labels.crs:
                print("Dimensions or CRS do not match, aligning the reference raster to match the features raster...")

                # Use match_rasters to align the reference raster
                self.src_labels = match_rasters(reference_path, self.image_path)

            # adjust dimensions based on offsets
            self.width = src_features.width - self.offset_left
            self.height = src_features.height - self.offset_top

            # precompute patch positions
            self.labeled_patches = []
            self.labeled_patches_count = 0

            for row in range(0, self.height - patch_size + 1, stride):
                for col in range(0, self.width - patch_size + 1, stride):

                    # extract the label for central pixel from the reference raster
                    center_row = row + self.offset_top + self.patch_size // 2
                    center_col = col + self.offset_left + self.patch_size // 2

                    central_window = rio.windows.Window(center_col, center_row, 1, 1)
                    label = self.src_labels.read(1, window=central_window).item()

                    # Only include patches with valid labels (non-background)
                    if label != self.background_label:
                        self.labeled_patches.append((row + self.offset_top, col + self.offset_left, label))
                        self.labeled_patches_count += 1

        print(f"Total ground truth patches generated: {self.labeled_patches_count}")

        # print the counts of unique labels
        labeled_patches_arr = np.array(self.labeled_patches)
        unique_labels, counts = np.unique(labeled_patches_arr[:, 2], return_counts=True)
        print("Unique Labels:", unique_labels)
        print("Counts:", counts)

    def __len__(self):
        return len(self.labeled_patches)

    def __getitem__(self, idx):
        row, col, label = self.labeled_patches[idx]

        # Extract the image patch
        with rio.open(self.image_path) as src_features:
            window = rio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch_features = src_features.read(window=window)

        # Return the patch and its label
        return torch.tensor(patch_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def generate_feature_patches_loader(image_path, patch_size, stride, batch_size, offset_left=0, offset_top=0, shuffle=False):
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

    features_dataset = FeaturePatchesDataset(image_path=image_path,
                                             patch_size=patch_size,
                                             stride=stride,
                                             offset_left=offset_left,
                                             offset_top=offset_top)

    return DataLoader(features_dataset, batch_size=batch_size, shuffle=shuffle)

def generate_labeled_patches_loader(image_path, reference_path, patch_size, stride, batch_size, offset_left=0, offset_top=0, background_label=0, shuffle=False):
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

    labeled_dataset = LabeledPatchesDataset(image_path=image_path,
                                            reference_path=reference_path,
                                            patch_size=patch_size,
                                            stride=stride,
                                            offset_left=offset_left,
                                            offset_top=offset_top,
                                            background_label=background_label)

    return DataLoader(labeled_dataset, batch_size=batch_size, shuffle=shuffle)

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