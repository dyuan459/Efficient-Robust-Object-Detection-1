import os
import json
from multiprocessing import Pool, cpu_count
from functools import partial


def load_annotations(annotation_name):
    path = f"annotations_trainval2017/annotations/instances_{annotation_name}.json"
    with open(path, "r") as f:
        return json.load(f)


def get_image_info(annotations):
    return {img["id"]: (img["width"], img["height"]) for img in annotations["images"]}


def label_image(img_name, image_dims, annotations, category_lim, mode, image_dir):
    if not img_name.endswith(".jpg"):
        return None

    image_id = int(img_name.replace(".jpg", ""))
    if image_id not in image_dims:
        return None

    img_width, img_height = image_dims[image_id]
    label_path = f"labels/{mode}/{str(image_id).zfill(12)}.txt"
    if os.path.exists(label_path):
        return f"{image_dir}/{img_name}\n"
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as fp:
        for ann in annotations["annotations"]:
            if ann["image_id"] != image_id or ann["category_id"] >= category_lim:
                continue

            x, y, w, h = ann["bbox"]
            cx, cy = (x + w / 2) / img_width, (y + h / 2) / img_height
            w, h = w / img_width, h / img_height

            label = ([image_id, ann["category_id"], cx, cy, w, h]
                     if mode == "train"
                     else [ann["category_id"], cx, cy, w, h])

            fp.write(" ".join(map(str, label)) + "\n")

    return f"{image_dir}/{img_name}\n"


def label_all_images(image_dir, annotation_name, category_lim, mode, num_workers=None):
    annotations = load_annotations(annotation_name)
    image_dims = get_image_info(annotations)
    image_files = os.listdir(image_dir)

    label_func = partial(
        label_image,
        image_dims=image_dims,
        annotations=annotations,
        category_lim=category_lim,
        mode=mode,
        image_dir=image_dir
    )

    num_workers = num_workers or cpu_count()
    print("labelling is starting...")

    with Pool(num_workers) as pool:
        results = pool.map(label_func, image_files)

    print("labelling has been done")
    list_file_path = f"data/COCO2017/{"5k.txt" if mode == "train" else "trainvalno5k.txt"}"
    with open(list_file_path, "w") as f:
        for r in results:
            if r:
                f.write(r)


if __name__ == "__main__":
    label_all_images("images/train", "train2017", 80, "train", num_workers=6)
    # label_all_images("images/valid", "val2017", 80, "valid", num_workers=4)
