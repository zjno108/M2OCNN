import glob
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from kornia.color import rgb_to_yuv
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(
        self,
        root,
        noise_level,
        count=None,
        transforms_1=None,
        transforms_2=None,
        unaligned=False,
    ):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level = noise_level

    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(
                np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)
            )

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(
                np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)
            )
        else:
            # if noise !=0, A and B make different transform
            item_A = self.transform1(
                np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)
            )
            item_B = self.transform1(
                np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)
            )

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        item_A = self.transform(
            np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)
        )
        if self.unaligned:
            item_B = self.transform(
                np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])
            )
        else:
            item_B = self.transform(
                np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)
            )
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def CreateDatasetSynthesis(phase, input_path, contrast1="t1", contrast2="t2"):
    root = os.path.join(input_path, phase)

    dataset = ImageDataset1(root, contrast1, contrast2)
    return dataset


class ImageDataset1(Dataset):
    def __init__(self, root, contrast1, contrast2):
        self.files_A = sorted(glob.glob(os.path.join(root, contrast1 + "_npy/*")))
        self.files_B = sorted(glob.glob(os.path.join(root, contrast2 + "_npy/*")))

    def __getitem__(self, index):
        item_A = np.load(self.files_A[index % len(self.files_A)]).astype(np.float32)

        item_A = transforms.ToTensor()(item_A)

        item_B = np.load(self.files_B[index % len(self.files_B)]).astype(np.float32)

        item_B = transforms.ToTensor()(item_B)

        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class HarvardMedImageRgb(Dataset):
    def __init__(self, image_pair_path_csv, transform=None):
        self.imgpairpath = np.loadtxt(image_pair_path_csv, dtype=str, delimiter=",")
        self.num_pairs = len(self.imgpairpath)
        self.transform = transform
        self.imagepairs = []
        for m1_path, m2_path in self.imgpairpath:
            img_m1 = read_image(m1_path)
            img_m2 = read_image(m2_path)
            if self.transform is not None:
                img_m1 = self.transform(img_m1)
                img_m2 = self.transform(img_m2)
            self.imagepairs.append([img_m1, img_m2])

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        imagepairs = self.imagepairs[idx]

        return imagepairs


class HarvardMedImageYuv(Dataset):
    def __init__(self, image_pair_path_csv, transform=None):
        self.imgpairpath = np.loadtxt(image_pair_path_csv, dtype=str, delimiter=",")
        self.num_pairs = len(self.imgpairpath)
        self.transform = transform
        self.imagepairs = []
        for m1_path, m2_path in self.imgpairpath:
            img_m1 = read_image(m1_path)
            img_m2 = read_image(m2_path)
            if self.transform is not None:
                img_m1 = self.transform(img_m1)
                img_m2 = self.transform(img_m2)
            img_m1 = rgb_to_yuv(img_m1)
            img_m2 = rgb_to_yuv(img_m2)
            self.imagepairs.append([img_m1, img_m2])

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        imagepairs = self.imagepairs[idx]
        return imagepairs


if __name__ == "__main__":
    import os

    import torch
    from torchvision.transforms import v2

    from dataset import HarvardMedImageYuv

    root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    trian_modalities = ["mri", "fmri"]
    test_modalities = ["mri", "pet"]
    trian_modalities_str = "_" + "_".join(trian_modalities)
    test_modalities_str = "_" + "_".join(test_modalities)
    train_csv_relpath = "csvfiles/train" + trian_modalities_str + ".csv"
    test_csv_relpath = "csvfiles/test" + test_modalities_str + ".csv"

    train_csv_abspath = os.path.join(os.getcwd(), train_csv_relpath)
    train_set = HarvardMedImageYuv(train_csv_abspath, transforms)

    test_csv_abspath = os.path.join(os.getcwd(), test_csv_relpath)
    test_set = HarvardMedImageYuv(test_csv_abspath, transforms)
