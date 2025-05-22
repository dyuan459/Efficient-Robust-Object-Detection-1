import json

with open(
        "../../OneDrive - The University of Western Ontario/Efficient-Robust-Object-Detection-master/Efficient-Robust-Object-Detection-master/annotations/instances_train2017.json", "r") as f:
    annotations = json.load(f)
    for r in annotations["categories"]:
        with open(
                "../../OneDrive - The University of Western Ontario/Efficient-Robust-Object-Detection-master/Efficient-Robust-Object-Detection-master/classes/class.txt", "a") as fp:
            fp.write(r["name"])
            fp.write("\n")
