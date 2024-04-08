import io
import json
import os
import argparse
import random
import shutil

ANN_DIR = "annotations"
CALIB_DIR = "calib_dataset"
EVAL_DIR = "eval_dataset"
ANN_JSON = ANN_DIR + "/instances_eval.json"


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
        "--num-of-images",
        type=int,
        default=100,
        help="# of images from each subfolders in CEPDOF to compose a dataset for calibration",
    )

    return ap


def verify_path(path):
    if not os.path.isdir(path):
        print(f"[ERROR] {path} not found")
        print("[INFO] Quiting early")
        quit()


def get_output_dir_paths(path):
    calib_dataset_path = os.path.join(path, CALIB_DIR)
    eval_dataset_path = os.path.join(path, EVAL_DIR)
    annotations_path = os.path.join(path, ANN_DIR)
    ann_json_path = os.path.join(path, ANN_JSON)

    # Create calibration dataset folder if not exists
    try:
        os.makedirs(calib_dataset_path)
        print(f"[INFO] created {CALIB_DIR} directory in {path}")
    except FileExistsError:
        pass

    # Create evaluation dataset folder if not exists
    try:
        os.makedirs(eval_dataset_path)
        print(f"[INFO] created {EVAL_DIR} directory in {path}")
    except FileExistsError:
        pass

    # Create annotations folder if not exists
    try:
        os.makedirs(annotations_path)
        print(f"[INFO] created {ANN_DIR} directory in {path}")
    except FileExistsError:
        pass

    # Create json file for annotations to evaluate quantized model
    if not (os.path.isfile(ann_json_path) and os.access(ann_json_path, os.R_OK)):
        with io.open(ann_json_path, "w") as json_file:
            json_file.write(json.dumps({}))
        print(f"[INFO] created empty json file: {ann_json_path}")

    return calib_dataset_path, eval_dataset_path, ann_json_path


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

    calib_dataset_path, eval_dataset_path, dest_json_path = get_output_dir_paths(
        args.output_dir
    )

    # Get all sub folders of CEPDOF except annotations
    subdirs = [
        dir
        for dir in os.scandir(args.input_dir)
        if dir.is_dir() and dir.name != "annotations"
    ]

    # Loop over each sub folder
    for subdir in subdirs:
        num_images = args.num_of_images

        # Get all images in the sub folder
        image_paths = os.listdir(subdir.path)

        if num_images > len(image_paths):
            num_images = len(image_paths)
            print(
                f"[WARNING] No. of images request is greater than the no. of images in {subdir.name}, reduced N to {len(image_paths)}."
            )

        # Get N unique images randomly for calibration
        calib_images = random.sample(os.listdir(subdir.path), k=num_images)

        # Copy picked images to calibration dataset folder
        for image in calib_images:
            shutil.copy2(os.path.join(subdir.path, image), calib_dataset_path)

        print(
            f"[INFO] {num_images} unique randomly picked images in {subdir.name} are copied to {calib_dataset_path}"
        )

        # Get N//2 unique images randomly for evaluation
        eval_images = random.sample(os.listdir(subdir.path), k=num_images // 2)

        # Copy picked images to evaluation dataset folder
        for image in eval_images:
            shutil.copy2(os.path.join(subdir.path, image), eval_dataset_path)

        print(
            f"[INFO] {num_images//2} unique randomly picked images in {subdir.name} are copied to {eval_dataset_path}"
        )

        source_json_path = os.path.join(args.input_dir, ANN_DIR, subdir.name) + ".json"

        with open(source_json_path, "r") as source_json, open(
            dest_json_path, "r"
        ) as dest_json:
            dest_dict = json.load(dest_json)

            source_dict = json.load(source_json)

            to_add_dict = get_images_and_annotations(eval_images, source_dict)

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

    print("[INFO] successfully created CEPDOF dataset for calibration and evaluation")
