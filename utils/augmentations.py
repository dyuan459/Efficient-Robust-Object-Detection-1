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