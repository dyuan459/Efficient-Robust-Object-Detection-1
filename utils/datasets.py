from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value): # not used
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset): # only used in detect for some reason and not others
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder which is filled later
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        # try:
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # Ignore warning if file is empty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        if os.path.exists(label_path) and os.stat(label_path).st_size > 0:
            boxes = np.loadtxt(label_path).reshape(-1, 6)
            print(f"Labels loaded for {img_path}: {boxes.shape}")
        else:
            print(f"No labels found for {img_path}")
            boxes = np.zeros((0, 6))  # No labels available
        # except Exception:
        #     print(f"Could not read label '{label_path}'.")
        #     return

        # -----------
        #  Transform
        # -----------

        #TODO: Here is the critical kicker, the transforms expect around 8 columns but we only give them 5!!!
        # label files seem roughly in the format: image_id, category_id, x_center, y_center, width, height, orig_height, orig_width
        if self.transform:
            # try:
            img, bb_targets = self.transform((img, boxes))
            # except Exception as e:
            #     print(f"Could not apply transform for {img_path} due to {e}.")
            #     return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            # print(f"look at boxes {boxes.shape}")
        #     print(f"index {i}")
        #     print(f"boxes {boxes}")
            boxes[:,0] = i
        #
        #
        bb_targets = torch.cat(bb_targets, 0)
        # hm... so image id gets replaced with sample index? Isn't the image id already a sample index?
        # okay so sample index will be important later

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

class ValidDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return
        # ---------
        #  Label
        # ---------
        # try:
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # Ignore warning if file is empty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        if os.path.exists(label_path) and os.stat(label_path).st_size > 0:
            boxes = np.loadtxt(label_path).reshape(-1, 5)
            print(f"Labels loaded for {img_path}: {boxes.shape}")
        else:
            print(f"No labels found for {img_path}")
            boxes = np.zeros((0, 5))  # No labels available
        # except Exception:
        #     print(f"Could not read label '{label_path}'.")
        #     return

        # -----------
        #  Transform
        # -----------

        #TODO: Here is the critical kicker, the transforms expect around 8 columns but we only give them 5!!!
        # label files seem roughly in the format: image_id, category_id, x_center, y_center, width, height, orig_height, orig_width
        if self.transform:
            # try:
            img, bb_targets = self.transform((img, boxes))
            # except Exception as e:
            #     print(f"Could not apply transform for {img_path} due to {e}.")
            #     return

        return img_path, img, bb_targets

    # def collate_fn(self, batch):
    #     self.batch_count += 1
    #
    #     # Drop invalid images
    #     batch = [data for data in batch if data is not None]
    #     paths, imgs, bb_targets = list(zip(*batch))
    #
    #     # Resize images to input shape
    #     if self.multiscale and self.batch_count % 10 == 0:
    #         self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
    #
    #     imgs = torch.stack([resize(img, self.img_size) for img in imgs])
    #
    #     # Insert sample index as new first column
    #     for i, boxes in enumerate(bb_targets):
    #         if boxes.size(0) > 0:  # Only if there are boxes
    #             # Create sample index column
    #             sample_indices = torch.full((boxes.size(0), 1), i, dtype=boxes.dtype)
    #             # Concatenate: [sample_index, original_data]
    #             boxes = torch.cat([sample_indices, boxes], dim=1)
    #             bb_targets[i] = boxes
    #
    #     bb_targets = torch.cat([b for b in bb_targets if b.size(0) > 0], 0)
    #
    #     return paths, imgs, bb_targets
    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        bb_targets = list(bb_targets)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            if boxes.size(0) > 0:  # Has boxes
                sample_indices = torch.full((boxes.size(0), 1), i, dtype=boxes.dtype)
                boxes = torch.cat([sample_indices, boxes], dim=1)
            else:  # No boxes - create empty tensor with correct shape
                boxes = torch.zeros((0, boxes.size(1) + 1), dtype=boxes.dtype)
            bb_targets[i] = boxes
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)