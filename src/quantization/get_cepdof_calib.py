"""
1. parse args: input folder path (CEPDOF), output folder (calib), how many frames in each subfolder
2. loop through subfolders in CEPDOF, randomly copy N frames into output folder from each subfolders
3. get all annotations 
"""

import io
import json
import os
import argparse
import random
import shutil

ANN_DIR = "annotations"
IMG_DIR = "images"
ANN_JSON = ANN_DIR + "/instances_calib.json"


def argparser_init():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=str,
        help="directory path to required input dataset for calibration",
    )

    ap.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="directory path to output calibration dataset",
    )

    ap.add_argument(
        "-n",
        "--images",
        type=int,
        default=50,
        help="# of images from each subfolders in CEPDOF",
    )

    return ap


def verify_path(path):
    if not os.path.isdir(path):
        print(f"[ERROR] {path} not found")
        print("[INFO] Quiting early")
        quit()


def get_output_dir_paths(path):
    # Create images folder if not exists
    try:
        os.makedirs(os.path.join(path, IMG_DIR))
        print(f"[INFO] created {IMG_DIR} directory in {path}")
    except FileExistsError:
        pass

    # Create annotations folder if not exists
    try:
        os.makedirs(os.path.join(path, ANN_DIR))
        print(f"[INFO] created {ANN_DIR} directory in {path}")
    except FileExistsError:
        pass

    # Create json file for annotations to evaluate quantized model
    dest_json_path = os.path.join(path, ANN_JSON)
    if not (os.path.isfile(dest_json_path) and os.access(dest_json_path, os.R_OK)):
        with io.open(dest_json_path, "w") as json_file:
            json_file.write(json.dumps({}))
        print(f"[INFO] created empty json file: {dest_json_path}")

    return os.path.join(path, IMG_DIR), dest_json_path


def get_images_and_annotations(images, source_dict):
    result_dict = {"annotations": [], "images": []}
    image_ids = []

    # Convert list to dictionary to speed up searching
    images_dict = {image["file_name"]: image for image in source_dict["images"]}

    for image_file in images:
        image_obj = images_dict.get(image_file, None)

        if image_obj is not None:
            # Store the id for later
            image_ids.append(image_obj["id"])
            result_dict["images"].append(image_obj)
        else:
            print(f"[WARNING] {image_file} object not found in the JSON file")

    # Get the annotations by image id
    for ann in source_dict["annotations"]:
        if ann["image_id"] in image_ids:
            result_dict["annotations"].append(ann)

    return result_dict


if __name__ == "__main__":
    args = argparser_init().parse_args()

    # Verify path in args
    verify_path(args.input_dir)
    verify_path(args.output_dir)

    dest_images_path, dest_json_path = get_output_dir_paths(args.output_dir)

    # Get all sub folders of CEPDOF except annotations
    subdirs = [
        dir
        for dir in os.scandir(args.input_dir)
        if dir.is_dir() and dir.name != "annotations"
    ]

    # Loop over each sub folder
    for subdir in subdirs:
        n = args.images

        # Get all images in the sub folder
        image_paths = os.listdir(subdir.path)

        if n > len(image_paths):
            n = len(image_paths)
            print(
                "[WARNING] No. of images request is greater than the no. of images in {path}, reduced N to match the no."
            )

        # Get N unique images randomly
        selected_images = random.sample(os.listdir(subdir.path), k=n)

        # Copy selected images to calibration dataset folder
        for image in selected_images:
            shutil.copy2(os.path.join(subdir.path, image), dest_images_path)

        print(
            f"[INFO] {n} randomly unique selected images in {subdir.name} are copied to {dest_images_path}"
        )

        source_json_path = os.path.join(args.input_dir, ANN_DIR, subdir.name) + ".json"

        with open(source_json_path, "r") as source_json, open(
            dest_json_path, "r"
        ) as dest_json:
            dest_dict = json.load(dest_json)

            source_dict = json.load(source_json)

            to_add_dict = get_images_and_annotations(selected_images, source_dict)

            if "images" not in dest_dict:
                dest_dict["images"] = to_add_dict["images"]
            else:
                dest_dict["images"] += to_add_dict["images"]

            if "annotations" not in dest_dict:
                dest_dict["annotations"] = to_add_dict["annotations"]
            else:
                dest_dict["annotations"] += to_add_dict["annotations"]

        with open(dest_json_path, "w") as dest_json:
            json.dump(dest_dict, dest_json)

    print("[INFO] successfully created CEPDOF dataset for calibration")

    """
    "annotations": [
        {
        "area": 131282.24863635,
        "bbox": [
            1442.4975,
            1364.545,
            256.9861,
            510.8535,
            -22.918329405326745
        ],
        "category_id": 1,
        "image_id": "Lunch3_000001",
        "iscrowd": 0,
        "segmentation": [],
        "person_id": 4
        },
    ]
    "images": [
        {
            "file_name": "Lunch3_000001.jpg",
            "id": "Lunch3_000001",
            "width": 2048,
            "height": 2048
        },
    ]
    """
