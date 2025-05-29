import json
import numpy as np

file_check = False

with open(
        "annotations_trainval2017\\annotations\\instances_train2017.json",
        "r"
) as f: # open annotations file to process
    annotations = json.load(f)
    with open(
            "classes\\class.txt",
            "w+"
    ) as fp:
        for r in annotations["categories"]: # setup names file
            if not file_check:
                first_char = fp.read(1)
                if not first_char: # if file is empty then start appending
                    file_check = True
                else: # if the file isn't empty then don't append anything and break out of loop
                    break
            fp.write(r["name"])
            fp.write("\n")
    file_check = False
    bboxes = [a["bbox"] for a in annotations["annotations"] if a["image_id"] == 9] #formatted [x_top_left, y_top_left, width, height]
    bboxes = np.array(bboxes)
    bboxes[:][0]+=(bboxes[:][2]/2) # get the absolute center x coords
    bboxes[:][1]+=(bboxes[:][3]/2) # get the absolute center y coords
    classid = np.array([c["category_id"] for c in annotations["annotations"] if c["image_id"] == 9])
    # right now I have absolute center x, abs center y, abs width, abs height
    img_width = np.array([b["width"] for b in annotations["images"] if b["id"] == 9])
    img_height = np.array([d["height"] for d in annotations["images"] if d["id"] == 9])
    print(bboxes)
    print(classid)
    print(img_width)
    print(img_height)
    # with open(
    #     "C:\\Users\\yyuan459\\PycharmProjects\\Efficient-Robust-Object-Detection\\labels\\train\\000000000009.txt", "w+"
    # ) as fp:
    #     if not file_check:
    #         first_char = fp.read(1)
    #         if not first_char:
    #             file_check = True
    #         else:
                # break

