import imgaug.augmenters as iaa
from utils.transforms import Resize, ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug, ConvertToArrays

import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from .utils import xywh2xyxy_np
import torchvision.transforms as transforms


class DefaultAug(ImgAug):
    def __init__(self):
        print("in DefaultAug")
        super().__init__(iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ]))
        print("default aug done")

    def __call__(self, data):
        print("da call")
        img, boxes = data
        print(f"da Pre-transform: {boxes[0] if boxes.size > 0 else 'empty'}")

        img, boxes = super().__call__(data)
        print(f"da success {boxes[0] if boxes.size > 0 else 'empty'}")
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

AUGMENTATION_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),  # Converts normalized to absolute pixels
    DefaultAug(),  # Requires absolute coordinates, mixes up image properties for training
    PadSquare(),  # Requires absolute coordinates, zooms into center of image
    RelativeLabels(),  # Converts back to normalized
    ToTensor(), # Converts to tensor with 7 values (cut off the original array's column 6-7 and append an anchor id)
])


class MyCompose(object):
    def __init__(self, transforms_):
        self.transforms = transforms_

    def __call__(self, data):
        img, tar = data
        for t in self.transforms:
            print("transforming", flush=True, end="")
            img, tar = t((img, tar))
        return img, tar


# TRANSFORM_TRAIN = MyCompose([
#     DefaultAug(),
#     PadSquare(),
#     RelativeLabels(),
#     ConvertToArrays(),
# ])

# TRANSFORM_VAL = MyCompose([
#     ConvertToArrays(),
#     # DefaultAug(),
#     PadSquare(),
#     ToTensor(),
#     RelativeLabels(),
#     Resize(416),
# ])

# TRANSFORM_VAL = MyCompose([
#     ConvertToArrays(),
#     PadSquare(),
#     RelativeLabels(),
#     ToTensor(),
#     Resize(416),
# ])