import json
import os

import numpy as np
#
#
# # image_id = 9
# def label_image(image_id, annotation_name): # assumes cocodataset annotation json starts with instances_
#     file_check = False
#     anno_path = "annotations_trainval2017\\annotations\\instances_" + annotation_name + ".json"
#     with open(
#             anno_path,
#             "r"
#     ) as f: # open annotations file to process
#         annotations = json.load(f)
#         with open(
#                 "classes\\class.txt",
#                 "w+"
#         ) as fp:
#             for r in annotations["categories"]: # setup names file
#                 if not file_check:
#                     first_char = fp.read(1)
#                     if not first_char: # if file is empty then start appending
#                         file_check = True
#                     else: # if the file isn't empty then don't append anything and break out of loop
#                         break
#                 fp.write(r["name"])
#                 fp.write("\n")
#         file_check = False
#         bboxes = [a["bbox"] for a in annotations["annotations"] if a["image_id"] == image_id] #formatted [x_top_left, y_top_left, width, height]
#         bboxes = np.array(bboxes)
#         for b in bboxes:
#             b[0] += b[2]/2 # get the absolute center x
#             b[1] += b[3]/2 # get the absolute center y
#
#         classid = np.array([c["category_id"] for c in annotations["annotations"] if c["image_id"] == image_id])
#         print(bboxes)
#         print(classid)
#         # right now I have absolute center x, abs center y, abs width, abs height
#         img_width = [b["width"] for b in annotations["images"] if b["id"] == image_id][0]
#         img_height = [d["height"] for d in annotations["images"] if d["id"] == image_id][0]
#
#         # now I want to convert coords to relative/normal
#         for b in bboxes:
#             b[0] /= img_width
#             b[2] /= img_width
#             b[1] /= img_height # transformations appear to expect relative dimensions too
#             b[3] /= img_height
#
#         # currently works fine so get rid of extraneous io
#         # print(bboxes)
#         # print(classid)
#         # print(img_width)
#         # print(img_height)
#
#         labels = []
#
#         for b, c in zip(bboxes,classid):
#             label = []
#             label.append(image_id)
#             label.append(c)
#             for value in b:
#                 label.append(value)
#             label.append(img_height)
#             label.append(img_width)
#             print(label)
#             labels.append(label)
#         print(labels)
#         str_im_id = str(image_id)
#         label_path =  "labels\\train\\" + str_im_id.zfill(13-len(str_im_id))+".txt"
#         with open(
#            label_path, "w+" # make sure this is general
#         ) as fp:
#             for l in labels:
#                 fp.write(", ".join(map(str, l))) # get rid of brackets while writing
#                 fp.write("\n")
#         #     if not file_check:
#         #         first_char = fp.read(1)
#         #         if not first_char:
#         #             file_check = True
#         #         else:
#                     # break
#     # for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
#     #     with open(label_path, "r") as f:
#     #         content = f.read()
#     #
#     #     # Remove commas and extra spaces
#     #     cleaned = content.replace(",", " ").replace("  ", " ")
#     #
#     #     with open(label_path, "w") as f:
#     #         f.write(cleaned)
#
# label_image(139, "val2017")
#
# # import glob
# # import os
# #
# # # Path to your labels directory
# # label_dir = "labels/"
#

# def label_image_training(image_id, annotation_name, category_lim):
#     anno_path = f"annotations_trainval2017/annotations/instances_{annotation_name}.json"
#
#     with open(anno_path, "r") as f:
#         annotations = json.load(f)
#
#         # Get image dimensions
#         image_info = next(i for i in annotations["images"] if i["id"] == image_id)
#         img_width, img_height = image_info["width"], image_info["height"]
#
#         # Create label file
#         str_im_id = str(image_id)
#         label_path = f"labels/train/{str_im_id.zfill(12)}.txt" # TODO: make this automated
#
#         with open(label_path, "w") as fp:
#             for ann in annotations["annotations"]:
#                 if ann["image_id"] != image_id or ann["category_id"] > category_lim:
#                     continue
#
#                 # Original COCO bbox format
#                 x, y, w, h = ann["bbox"]
#
#                 # Create 8-value label
#                 label = [
#                     image_id,  # 0
#                     ann["category_id"],  # 1
#                     (x + w / 2) / img_width,  # 2
#                     (y + h / 2) / img_height,  # 3
#                     w / img_width,  # 4
#                     h / img_height,  # 5
#                     img_height,  # 6
#                     img_width  # 7
#                 ]
#
#                 fp.write(" ".join(map(str, label)) + "\n")
#
# # label_image(9, "train2017")


def label_image_con(valtrain): # remember only either "valid" or "train"
    def label_image(image_id, annotation_name, category_lim):

        anno_path = f"annotations_trainval2017/annotations/instances_{annotation_name}.json"

        with open(anno_path, "r") as f:
            annotations = json.load(f)

            # Get image dimensions
            image_info = next(i for i in annotations["images"] if i["id"] == image_id)
            img_width, img_height = image_info["width"], image_info["height"]

            # Create label file
            str_im_id = str(image_id)
            label_path = f"labels/{valtrain}/{str_im_id.zfill(12)}.txt"  # TODO: make this automated

            with open(label_path, "w") as fp:
                for ann in annotations["annotations"]:
                    if ann["image_id"] != image_id or ann["category_id"] > category_lim:
                        continue

                    # Original COCO bbox format
                    x, y, w, h = ann["bbox"]
                    if valtrain == "train":
                        # Create 8-value label
                        label = [
                            image_id,  # 0
                            ann["category_id"],  # 1
                            (x + w / 2) / img_width,  # 2
                            (y + h / 2) / img_height,  # 3
                            w / img_width,  # 4
                            h / img_height,  # 5
                            img_height,  # 6
                            img_width  # 7
                        ]
                    elif valtrain == "valid":
                        label = [
                            ann["category_id"],  # 0
                            (x + w / 2) / img_width,  # 1
                            (y + h / 2) / img_height,  # 2
                            w / img_width,  # 3
                            h / img_height,  # 4
                        ]

                    fp.write(" ".join(map(str, label)) + "\n")
    return label_image

train_label = label_image_con("train")
val_label = label_image_con("valid")

imgs = os.listdir("images/train")
with open("data/COCO2017/5k.txt", "w") as f:
    for img in imgs:
        print(img)
        name = img.replace(".jpg", "")
        num = int(name)
        train_label(num, "train2017", 80)
        f.write("images/train/" + img + "\n")

imgs = os.listdir("images/valid")
with open("data/COCO2017/trainvalno5k.txt", "w") as f:
    for img in imgs:
        print(img)
        name = img.replace(".jpg", "")
        num = int(name)
        val_label(num, "val2017", 80)
        f.write("images/train/" + img + "\n")