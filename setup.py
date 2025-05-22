import json

file_check = False

with open(
        "C:\\Users\\Danry\\PycharmProjects\\Efficient-Robust-Object-Detection\\annotations_trainval2017\\annotations\\instances_train2017.json",
        "r"
) as f: # open annotations file to process
    annotations = json.load(f)
    for r in annotations["categories"]: # setup names file
        with open(
                "C:\\Users\\Danry\\PycharmProjects\\Efficient-Robust-Object-Detection\\classes\\class.txt",
                "a+"
        ) as fp:
            if not file_check:
                first_char = fp.read(1)
                if not first_char: # if file is empty then start appending
                    file_check = True
                else: # if the file isn't empty then don't append anything and break out of loop
                    break
            fp.write(r["name"])
            fp.write("\n")

    train = [a["bbox"] for a in annotations["annotations"] if a["image_id"] == 9]
    print(train)

