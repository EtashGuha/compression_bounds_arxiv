"""Dataset and DataModule for the FMOW dataset."""

# Imports Python builtins.
import os.path as osp

# Imports Python packages.
import numpy as np
from transformers import BertTokenizer
import wilds

# Imports PyTorch packages.
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# Imports groundzero packages.
from groundzero.datamodules.dataset import Dataset
from groundzero.datamodules.datamodule import DataModule
from groundzero.datamodules.disagreement import Disagreement
from groundzero.utils import to_np


class FMOWDataset(Dataset):
    """Dataset for the FMOW dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self):
        pass

    def load_data(self):
        dataset = wilds.get_dataset(dataset="fmow", download=True, root_dir=self.root)

        column_names = dataset.metadata_fields
        spurious_cols = column_names.index("region")
        spurious = to_np(dataset._metadata_array[:, spurious_cols])

        self.data = np.asarray([osp.join(self.root, "fmow_v1.1", "images", f"rgb_img_{idx}.png") for idx in dataset.full_idxs])
        self.targets = dataset.y_array

        # Spurious 5 represents "other" locations (unused).
        self.groups = [
            np.arange(len(self.targets)),
            np.argwhere(spurious == 0).squeeze(),
            np.argwhere(spurious == 1).squeeze(),
            np.argwhere(spurious == 2).squeeze(),
            np.argwhere(spurious == 3).squeeze(),
            np.argwhere(spurious == 4).squeeze(),
        ]
        
        # Splits 1 and 2 are in-distribution val and test (unused).
        split = dataset._split_array
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 3).flatten()
        self.test_indices = np.argwhere(split == 4).flatten()

class FMOW(DataModule):
    """DataModule for the FMOW dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, FMOWDataset, 62, **kwargs)

    def augmented_transforms(self):
        transform = Compose([
            RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

    def default_transforms(self):
        transform = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

class FMOWDisagreement(FMOW, Disagreement):
    """DataModule for the FMOWDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

