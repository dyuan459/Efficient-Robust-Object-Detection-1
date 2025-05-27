import json

file_check = False

with open(
        "annotations_trainval2017\\annotations\\instances_train2017.json",
        "r"
) as f: # open annotations file to process
    annotations = json.load(f)
    with open(
            "classes\\class.txt",
            "a+"
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
    train = [a["bbox"] for a in annotations["annotations"] if a["image_id"] == 9]
    print(train)
    # with open(
    #     "C:\\Users\\yyuan459\\PycharmProjects\\Efficient-Robust-Object-Detection\\labels\\train\\000000000009.txt", "w+"
    # ) as fp:
    #     if not file_check:
    #         first_char = fp.read(1)
    #         if not first_char:
    #             file_check = True
    #         else:
                # break

