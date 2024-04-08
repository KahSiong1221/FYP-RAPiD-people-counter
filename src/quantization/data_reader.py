from onnxruntime.quantization import CalibrationDataReader
import onnxruntime
import numpy as np
import cv2
import PIL

import sys
import os

sys.path.insert(1, "..")

from RAPiD.utils import utils as rapid_utils


def parse_annotations(filename):
    import json

    annotations = {}
    with open(filename, "r") as f:
        annotations = json.load(f)

    img_name_to_img_id = {}
    for image in annotations["images"]:
        file_name = image["file_name"]
        img_name_to_img_id[file_name] = image["id"]

    return img_name_to_img_id


def rapid_preprocess_func(images_folder, input_size, start_index=0, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter input_size: image size in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images (nchw_data_list, filename_list, image_size_list)
    """

    def _preprocess_rapid(img, input_size):
        """Preprocess an image before TRT RAPiD inferencing.
        # Args
            img: int8 numpy array of shape (img_h, img_w, 3)
            input_size: size of the img such as 608, 1024, etc.
        # Returns
            preprocessed img: float32 numpy array of shape (3, H, W)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(img_rgb)

        img_resized, _, _ = rapid_utils.rect_to_square(img_pil, None, input_size)

        img_input = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0

        return img_input

    image_names = os.listdir(images_folder)
    if start_index >= len(image_names):
        return np.asanyarray([]), np.asanyarray([]), np.asanyarray([])
    elif size_limit > 0 and len(image_names) >= size_limit:
        end_index = start_index + size_limit
        if end_index > len(image_names):
            end_index = len(image_names)

        batch_filenames = [image_names[i] for i in range(start_index, end_index)]
    else:
        batch_filenames = image_names

    unconcatenated_batch_data = []
    image_size_list = []

    print(
        f"[INFO] calibration pre-processing {len(batch_filenames)} files: {batch_filenames}"
    )

    for image_name in batch_filenames:
        image_filepath = os.path.join(images_folder, image_name)
        img = cv2.imread(image_filepath)
        image_data = _preprocess_rapid(img, input_size)
        image_data = np.ascontiguousarray(image_data)
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        image_size_list.append(img.shape[0:2])  # img.shape is h, w, c

    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data, batch_filenames, image_size_list


class ObejctDetectionDataReader(CalibrationDataReader):
    def __init__(self, model_path="int8_model.onnx"):
        self.model_path = model_path
        self.preprocess_flag = None
        self.start_index = 0
        self.end_index = 0
        self.stride = 1
        self.batch_size = 1
        self.enum_data_dicts = iter([])
        self.input_name = None
        self.get_input_name()

    def get_batch_size(self):
        return self.batch_size

    def get_input_name(self):
        if self.input_name:
            return
        session = onnxruntime.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = session.get_inputs()[0].name


class RapidDataReader(ObejctDetectionDataReader):
    """
    A subclass of CalibrationDataReader specifically designed for handling
    image data for calibration in RAPiD. This reader loads, preprocesses,
    and provides images for model calibration.
    """

    def __init__(
        self,
        calibration_dataset,
        input_size=1024,
        start_index=0,
        end_index=0,
        stride=1,
        batch_size=1,
        model_path="augmented_model.onnx",
        is_evaluation=False,
        annotations="./CEPDOF_calib/annotations/instances_eval.json",
        preprocess_func=rapid_preprocess_func,
    ):
        ObejctDetectionDataReader.__init__(self, model_path)
        self.image_folder = calibration_dataset
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])  # an interator over the image paths
        self.input_size = input_size
        self.start_index = start_index
        self.end_index = (
            len(os.listdir(calibration_dataset)) if end_index == 0 else end_index
        )
        self.stride = stride if stride >= 1 else 1  # stride must > 0
        self.batch_size = batch_size
        self.is_evaluation = is_evaluation
        self.img_name_to_img_id = parse_annotations(annotations)
        self.preprocess_func = preprocess_func

    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            print(f"[DEBUG] {iter_data}")
            return iter_data

        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        input_size = self.input_size
        input_name = self.input_name
        nchw_data_list, filename_list, image_size_list = self.preprocess_func(
            self.image_folder, input_size, self.start_index, self.stride
        )

        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id
            for i in range(len(nchw_data_list)):
                nchw_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append(
                    {
                        input_name: nchw_data,
                        "image_id": img_name_to_img_id[file_name],
                        "image_size": image_size_list[i],
                    }
                )

        else:
            for i in range(len(nchw_data_list)):
                nchw_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nchw_data})

        return data

    def load_batches(self):
        input_size = self.input_size
        stride = self.stride
        batch_size = self.batch_size
        input_name = self.input_name

        batches = []
        for index in range(0, stride, batch_size):
            start_index = self.start_index + index
            print(f"[INFO] load batch from index {start_index} ...")
            nchw_data_list, filename_list, image_size_list = self.preprocess_func(
                self.image_folder, input_size, start_index, batch_size
            )

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            image_id_batch = []
            if self.is_evaluation:
                img_name_to_img_id = self.img_name_to_img_id
                for i in range(len(nchw_data_list)):
                    nchw_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nchw_data)
                    img_name = filename_list[i]
                    image_id = img_name_to_img_id[img_name]
                    image_id_batch.append(image_id)
                batch_data = np.concatenate(
                    np.expand_dims(nchw_data_batch, axis=0), axis=0
                )
                batch_id = np.concatenate(
                    np.expand_dims(image_id_batch, axis=0), axis=0
                )
                print(f"[INFO] batch data shape: {batch_data.shape}")
                data = {
                    input_name: batch_data,
                    "image_size": image_size_list,
                    "image_id": batch_id,
                }
            else:
                for i in range(len(nchw_data_list)):
                    nchw_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nchw_data)
                batch_data = np.concatenate(
                    np.expand_dims(nchw_data_batch, axis=0), axis=0
                )
                print(batch_data.shape)
                data = {input_name: batch_data}

            batches.append(data)

        return batches
