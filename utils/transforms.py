import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms

import sys


# * modified transforms
# class ImgAug(object):
#     def __init__(self, augmentations):
#         self.augmentations = augmentations
#
#     def __call__(self, data):
#         img, boxes = data
#         # np.set_printoptions(linewidth=500)
#         # np.set_printoptions(suppress=True)
#         # print("before padsquare", boxes.shape)
#         # for box in boxes: print(box)
#
#         print("ia squeeze start")
#         if boxes.size != 0:
#             if boxes.ndim > 2:
#                 boxes = boxes.squeeze()
#             if boxes.ndim == 1:
#                 boxes = boxes.unsqueeze(0)
#             print("ia squeeze end")
#
#             # Convert xywh to xyxy
#             boxes = np.array(boxes)
#             image_size = boxes[:, 6:]
#             category_labels = boxes[:, 1:2] # sus, is this treating python as 1-indexed?
#             boxes[:, 2:6] = xywh2xyxy_np(boxes[:, 2:6])
#             # print("1", boxes.shape)
#
#             # Convert bounding boxes to imgaug
#             bounding_boxes = BoundingBoxesOnImage(
#                 [BoundingBox(*box[2:6], label=box[0:1]) for box in boxes],
#                 shape=img.shape)
#             print("ia aug start", flush=True)
#             # Apply augmentations
#             img, bounding_boxes = self.augmentations(
#                 image=img,
#                 bounding_boxes=bounding_boxes)
#             print("ia aug end", flush=True)
#             # Clip out of image boxes
#             bounding_boxes = bounding_boxes.clip_out_of_image()
#
#
#             # Convert bounding boxes back to numpy
#             boxes = np.zeros((len(bounding_boxes), 6))
#             for box_idx, box in enumerate(bounding_boxes):
#                 # Extract coordinates for unpadded + unscaled image
#                 x1 = box.x1
#                 y1 = box.y1
#                 x2 = box.x2
#                 y2 = box.y2
#
#                 # Returns (x, y, w, h)
#                 boxes[box_idx, 0:1] = box.label
#                 boxes[box_idx, 2] = ((x1 + x2) / 2)
#                 boxes[box_idx, 3] = ((y1 + y2) / 2)
#                 boxes[box_idx, 4] = (x2 - x1)
#                 boxes[box_idx, 5] = (y2 - y1)
#
#             print(image_size.shape, boxes.shape)
#             boxes = np.hstack((boxes, image_size)) # append the image size back on
#             boxes[:, 1:2] = category_labels
#             if boxes.shape[0] == 0: boxes = boxes[:, 0]
#         return img, boxes
class ImgAug(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        img, boxes = data
        if boxes.size == 0:
            return img, boxes

        # Handle single box case
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        print("ia meta pre", boxes)
        # Extract metadata
        # image_ids = boxes[:, 0].copy()  # 0: image_id
        category_ids = boxes[:, 0].copy()  # 0: category_id
        # orig_sizes = boxes[:, 5:7].copy()  # 6-7: orig_height, orig_width
        # Extract bbox values (columns 1-5)
        bbox_values = boxes[:, 1:6].copy()

        # Convert to xyxy format for augmentation
        bbox_values_xyxy = xywh2xyxy_np(bbox_values)
        print("ia bboxes")
        # Create bounding boxes
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
             for box in bbox_values_xyxy],
            shape=img.shape
        )
        print("ia aug")
        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes
        )
        print("ia aug done")
        # Clip boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()
        print("ia conversion")
        # Convert back to xywh format
        new_boxes = []
        kept_indices = []
        for i, box in enumerate(bounding_boxes):
            if box.is_fully_within_image(img): # make sure to check if box is dropped
                x_center = (box.x1 + box.x2) / 2
                y_center = (box.y1 + box.y2) / 2
                width = box.x2 - box.x1
                height = box.y2 - box.y1
                new_boxes.append([x_center, y_center, width, height])
                kept_indices.append(i)
        print("ia meta combine")
        # Recombine with metadata
        if new_boxes:
            # print(new_boxes)
            bbox_values = np.array(new_boxes)
            print("ia bbox", bbox_values.shape)
            # print("ia ii", image_ids.shape)
            print("ia ci", category_ids.shape)
            # print("ia orig", orig_sizes.shape)
            # Reconstruct full 8-value format
            boxes = np.column_stack([
                # image_ids[kept_indices],  # Filtered metadata
                category_ids[kept_indices],
                bbox_values,
                # orig_sizes[kept_indices]
            ])
            print("ia box", boxes)
        else:
            boxes = np.zeros((0, 5))
        print("ia success", end="", flush=True)
        return img, boxes


# class RelativeLabels(object):
#     def __init__(self, ):
#         pass
#
#     def __call__(self, img, boxes):
#         if boxes.shape[0] == 0: return img, boxes
#         _, h, w = img.shape # torch.Size([3, 640, 640])
#         if boxes.ndim > 2:
#             boxes = boxes.squeeze()
#         if boxes.ndim < 2:
#             boxes = boxes.unsqueeze(0)
#         boxes[:, [2, 4]] /= w
#         boxes[:, [3, 5]] /= h
#         return img, boxes
#
#
# class AbsoluteLabels(object):
#     """
#     callable class that transforms relative labels to absolute labels by scaling them by width and height
#     """
#     def __init__(self, ):
#         pass
#
#     def __call__(self, data):
#         img, boxes = data # expects tuple data input
#         _, h, w = img.shape # image shape probably has 3 dimensions to it: channels?, height, and width
## both item 3 and 5 have something to do with x
        # boxes[:, [2, 4]] *= w
        # boxes[:, [3, 5]] *= h
# both item 4 and 6 have something to do with y
#         # this formatting should mean that the expected input for boxes is an array of unspecific arrays and that columns 2 and 4 go to width and 3 and 5 go to height
#         return img, boxes

class AbsoluteLabels(object):
    def __call__(self, data):
        img, boxes = data
        print(f"al Pre-transform: {boxes[0] if boxes.size > 0 else 'empty'}")
        if boxes.size == 0:
            return img, boxes

        h, w, _ = img.shape  # Image is (H, W, C) numpy array
        print("shape")
        boxes = boxes.copy()
        print("copy")

        # Only convert bbox values (columns 2-5)
        boxes[:, [1, 3]] *= w  # x_center and width
        boxes[:, [2, 4]] *= h  # y_center and height
        print("transformed")
        return img, boxes


class RelativeLabels(object):
    def __call__(self, data):
        img, boxes = data
        print(f"Pre-transform: {boxes[0] if boxes.size > 0 else 'empty'}")
        if boxes.size == 0:
            return img, boxes

        h, w, _ = img.shape
        boxes = boxes.copy()

        # Only convert bbox values (columns 2-5)
        boxes[:, [1, 3]] /= w  # x_center and width
        boxes[:, [2, 4]] /= h  # y_center and height
        return img, boxes


# class PadSquare(ImgAug):
#     def __init__(self, ):
#         self.augmentations = iaa.Sequential([
#             iaa.PadToAspectRatio(
#                 1.0,
#                 position="center-center").to_deterministic()
#         ])

class PadSquare(ImgAug):
    def __init__(self):
        print("ps")
        super().__init__(iaa.Sequential([
            iaa.PadToAspectRatio(1.0, position="center-center").to_deterministic()
        ]))
        print("ps done")

    def __call__(self, data):
        img, boxes = data
        print(f"ps Pre-transform: {boxes[0] if boxes.size > 0 else 'empty'}", end="", flush=True)
        # if boxes.size > 0:
        #     # Save metadata
        #     metadata = boxes[:, :6].copy()  # First 6 columns
        #     orig_sizes = boxes[:, 6:8].copy()
        #     bbox_values = boxes[:, 2:6].copy()
        #     print("ps metadata",end="", flush=True)
        # else:
        #     metadata = np.zeros((0, 6))
        #     orig_sizes = np.zeros((0, 2))
        #     bbox_values = np.zeros((0, 4))

        # Apply padding to image and bbox_values only
        img, bbox_values = super().__call__((img, boxes))
        print("ps super", end="", flush=True)
        # Reconstruct full boxes
        if bbox_values.size > 0:
            # boxes = np.hstack([
            #     metadata[:, :2],  # image_id, category_id
            #     bbox_values,
            #     metadata[:, 4:6],  # ? (depends on your format)
            #     orig_sizes
            # ])
            print("ps recon", end="", flush=True)
        else:
            boxes = np.zeros((0, 5))
        print("ps success", end="", flush=True)
        return img, boxes


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        img = transforms.ToTensor()(img)
        boxes = boxes[:, 0:6]
        boxes = torch.tensor(boxes)
        return img, boxes

# class ToTensor(object):
#     def __call__(self, data):
#         img, boxes = data
#         print(f"tt Pre-transform: {boxes[0] if boxes.size > 0 else 'empty'}")
#         # Convert image
#         img = transforms.ToTensor()(img)
#
#         # Convert boxes
#         if isinstance(boxes, np.ndarray):
#             boxes = torch.from_numpy(boxes).float()
#         elif not isinstance(boxes, torch.Tensor):
#             boxes = torch.zeros((0, 8))
#         print("tt success")
#         return img, boxes


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


# * original transforms, not in use
class ImgAugEval(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = ((x1 + x2) / 2)
            boxes[box_idx, 2] = ((y1 + y2) / 2)
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes


class ResizeEval(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


class RelativeLabelsEval(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


class AbsoluteLabelsEval(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


class PadSquareEval(ImgAugEval):
    def __init__(self):
        super().__init__(iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ]))

    def __call__(self, data):
        img, bbox_values = super().__call__(data)
        print("ps Pre-transform", end="", flush=True)
        return img, bbox_values


class ToTensorEval(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


class ConvertToArrays():
    def __init__(self, ):
        pass

    def __call__(self, img, boxes):
        transformed_samples = []
        width, height = img.size
        for box in boxes:
            # Extract the required fields and create the array
            transformed_sample = np.array([
                box['image_id'],
                box['category_id'],
                *box['bbox'],
                height,  # put original image shape in the right columns of target (h,w)
                width,
            ])
            transformed_samples.append(transformed_sample)
        # np.set_printoptions(suppress=True)
        # print("original")
        # for box in transformed_samples: print(box)
        return np.array(img), np.array(transformed_samples)


DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabelsEval(),
    PadSquareEval(),
    RelativeLabelsEval(),
    ToTensorEval(),
])
