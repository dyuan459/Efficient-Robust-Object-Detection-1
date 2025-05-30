import imgaug.augmenters as iaa
from utils.transforms import Resize, ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug, ConvertToArrays

import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms

class DefaultAug(ImgAug):
    def __init__(self, ):
        print("in DefaultAug")
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])
        print("default aug done")
    def __call__(self, data):
        print("da call")
        img, boxes = data
        print(f"da Pre-transform: {boxes[0] if boxes.size > 0 else 'empty'}")
        if boxes.size == 0:
            return img, boxes

        # Handle single box case
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        # Extract metadata
        image_ids = boxes[:, 0].copy()  # 0: image_id
        category_ids = boxes[:, 1].copy()  # 1: category_id
        orig_sizes = boxes[:, 6:8].copy()  # 6-7: orig_height, orig_width

        # Extract bbox values (columns 2-5)
        bbox_values = boxes[:, 2:6].copy()

        # Convert to xyxy format for augmentation
        bbox_values_xyxy = xywh2xyxy_np(bbox_values)

        # Create bounding boxes
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
             for box in bbox_values_xyxy],
            shape=img.shape
        )

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes
        )

        # Clip boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert back to xywh format
        new_boxes = []
        for box in bounding_boxes:
            x_center = (box.x1 + box.x2) / 2
            y_center = (box.y1 + box.y2) / 2
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            new_boxes.append([x_center, y_center, width, height])

        # Recombine with metadata
        if new_boxes:
            bbox_values = np.array(new_boxes)
            # Reconstruct full 8-value format
            boxes = np.column_stack([
                image_ids,
                category_ids,
                bbox_values,
                orig_sizes
            ])
        else:
            boxes = np.zeros((0, 8))
        print("da success")
        return img, boxes


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])


# AUGMENTATION_TRANSFORMS = transforms.Compose([
#     AbsoluteLabels(),     # Converts normalized to absolute
#     DefaultAug(),         # Requires absolute coords
#     PadSquare(),          # Requires absolute coords
#     RelativeLabels(),     # Converts back to normalized
#     ToTensor(),
# ])

AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),     # Converts normalized to absolute pixels
    DefaultAug(),         # Requires absolute coordinates
    PadSquare(),          # Requires absolute coordinates
    RelativeLabels(),     # Converts back to normalized
    ToTensor(),
])



class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar
    
TRANSFORM_TRAIN = MyCompose([
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ConvertToArrays(),
])

# TRANSFORM_VAL = MyCompose([
#     ConvertToArrays(),
#     # DefaultAug(),
#     PadSquare(),
#     ToTensor(),
#     RelativeLabels(),
#     Resize(416),
# ])

TRANSFORM_VAL = MyCompose([
    ConvertToArrays(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    Resize(416),
])