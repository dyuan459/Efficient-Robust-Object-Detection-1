import json
import numpy as np

file_check = False
image_id = 9

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
    bboxes = [a["bbox"] for a in annotations["annotations"] if a["image_id"] == image_id] #formatted [x_top_left, y_top_left, width, height]
    bboxes = np.array(bboxes)
    for b in bboxes:
        b[0] += b[2]/2 # get the absolute center x
        b[1] += b[3]/2 # get the absolute center y

    classid = np.array([c["category_id"] for c in annotations["annotations"] if c["image_id"] == image_id])

    # right now I have absolute center x, abs center y, abs width, abs height
    img_width = [b["width"] for b in annotations["images"] if b["id"] == image_id][0]
    img_height = [d["height"] for d in annotations["images"] if d["id"] == image_id][0]

    # now I want to convert coords to relative/normal
    for b in bboxes:
        b[0] /= img_width
        b[2] /= img_width
        b[1] /= img_height # transformations appear to expect relative dimensions too
        b[3] /= img_height

    # currently works fine so get rid of extraneous io
    # print(bboxes)
    # print(classid)
    # print(img_width)
    # print(img_height)

    labels = []

    for b, c in zip(bboxes,classid):
        label = []
        label.append(image_id)
        label.append(c)
        for value in b:
            label.append(value)
        label.append(img_height)
        label.append(img_width)
        print(label)
        labels.append(label)
    print(labels)
    str_im_id = str(image_id)

    with open(
        "labels\\train\\" + str_im_id.zfill(13-len(str_im_id))+".txt", "w+" # make sure this is general
    ) as fp:
        for l in labels:
            fp.write(", ".join(map(str, l))) # get rid of brackets while writing
            fp.write("\n")
    #     if not file_check:
    #         first_char = fp.read(1)
    #         if not first_char:
    #             file_check = True
    #         else:
                # break

