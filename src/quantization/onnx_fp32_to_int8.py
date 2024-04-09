import argparse
import os

from onnxruntime.quantization import (
    create_calibrator,
    CalibrationMethod,
    write_calibration_table,
)

from data_reader import RapidDataReader

FRAME_SIZE = 1024
STRIDE = 5
BATCH_SIZE = 1
TRT_CACHE_DIR = "./trt_engine_cache"

def argparser_init():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="path to required pre-trained ONNX model",
    )

    ap.add_argument(
        "-t",
        "--augmented-model",
        default="augmented_model.onnx",
        type=str,
        help="path to temporary ONNX model used during calibration",
    )

    ap.add_argument(
        "-c",
        "--calib-dataset",
        required=True,
        type=str,
        help="path to required dataset for calibration",
    )

    ap.add_argument(
        "-e",
        "--eval-dataset",
        type=str,
        help="path to optional dataset for evaluation",
    )

    ap.add_argument(
        "-a",
        "--anns-file",
        type=str,
        help="path to optional annotations json file for evaluation",
    )

    ap.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="run evaluation on quantized model",
    )

    return ap


def get_calibration_table(model_path, augmented_model_path, calibration_dataset):
    calibrator = create_calibrator(
        model=model_path,
        op_types_to_calibrate=None,
        augmented_model_path=augmented_model_path,
        calibrate_method=CalibrationMethod.Entropy,
        use_external_data_format=False,  # True if model size >= 2GB
        extra_options={
            "symmetric": True
        },  # TensorRT requires symmetric quantization scheme
    )
    calibrator.set_execution_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    total_data_size = len(os.listdir(calibration_dataset))
    start_index = 0

    for i in range(0, total_data_size, STRIDE):
        data_reader = RapidDataReader(
            calibration_dataset=calibration_dataset,
            input_size=FRAME_SIZE,
            start_index=start_index,
            end_index=start_index + STRIDE,
            stride=STRIDE,
            batch_size=BATCH_SIZE,
            model_path=augmented_model_path,
        )
        calibrator.collect_data(data_reader)
        start_index += STRIDE

    """
        # write_calibration_table(calibrator.compute_range())
        The original code from example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/object_detection/trt/yolov3/e2e_user_yolov3_example.py
        but it is not working because they replaced compute_range() with 
        compute_data() in the calibrator API but didn't update the example
        and write_calibration_table(), it is using old data structure.
        So we have to manually change the new data structure back to the old one,
        data is returned by calibrator.compute_data().

        Old {tensor node name: [min_range, max_range]}, example:
        compute_range = {"L": [0.1, 0.1], "Q": [0.1, 0.1], ......}

        New {tensor node name: TensorData([min_range, max_range])}, example:
        compute_range = {"L": [0.1, 0.1], "Q": [0.1, 0.1], ......}
    """

    calib_ranges = parse_tensors_data(calibrator.compute_data())
    write_calibration_table(calib_ranges, dir=TRT_CACHE_DIR)
    print("[INFO] calibration table generated and saved")


def get_prediction_evaluation(model_path, eval_dataset, anns_file, providers):
    # TODO
    pass


def parse_tensors_data(tensors_data):
    """Parse TensorsData to a dictionary for write_calibration_table()"""
    calib_ranges = {}

    for tensor_node, tensor_data in tensors_data.data.items():
        min_range, max_range = tensor_data.range_value
        calib_ranges[tensor_node] = (float(min_range.item()), float(max_range.item()))

    return calib_ranges


if __name__ == "__main__":
    args = argparser_init().parse_args()

    # Create TensorRT engine cache folder if not exists
    try:
        os.makedirs(TRT_CACHE_DIR)
        print(f"[INFO] created {TRT_CACHE_DIR} folder")
    except FileExistsError:
        pass

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = (
        "calibration.flatbuffers"  # Calibration table name
    )
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    execution_provider = ["TensorrtExecutionProvider"]

    get_calibration_table(args.model, args.augmented_model, args.calib_dataset)

    if args.eval:
        if args.eval_dataset is None:
            print("[ERROR] path to evaluation dataset is mandatory if eval flag is set")
            print("[INFO] terminate evaluation, quiting early")
            quit()

        if args.anns_file is None:
            print(
                "[ERROR] path to annotations json file is mandatory if eval flag is set"
            )
            print("[INFO] terminate evaluation, quiting early")
            quit()

        get_prediction_evaluation(
            args.model, args.eval_dataset, args.anns_file, execution_provider
        )
