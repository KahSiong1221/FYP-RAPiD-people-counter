import argparse


def create_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument(
        "-w",
        "--weights",
        required=True,
        type=str,
        help="path to required pre-trained network model/weights",
    )

    ap.add_argument(
        "-e",
        "--engine-type",
        required=True,
        choices=["pytorch", "onnx", "tensorrt"],
        help="AI inferencing via pytorch, onnx or tensorrt",
    )

    ap.add_argument(
        "-p",
        "--execution-provider",
        required=True,
        choices=["cpu", "cuda", "tensorrt"],
        help="run object detection on cpu, cuda or tensorrt",
    )

    ap.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to optional input video file",
    )

    ap.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to optional output video file",
    )

    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.3,
        help="minimum probability to filter weak detections",
    )

    ap.add_argument(
        "-s",
        "--skip-frames",
        type=int,
        default=30,
        help="# of skip frames between detections",
    )

    ap.add_argument(
        "-f",
        "--framesize",
        type=int,
        default=1024,
        help="frame size of input video/webcam",
    )

    ap.add_argument(
        "-d",
        "--display",
        action="store_true",
        default=False,
        help="display processed frames on screen",
    )

    ap.add_argument(
        "-r",
        "--recursion",
        type=int,
        default=1,
        help="# of iterations to process the input",
    )

    ap.add_argument(
        "--on-device",
        action="store_true",
        default=False,
        help="enable IOBinding, allowing copy inputs onto GPUs and pre-allocate memory for outputs prior the inference",
    )

    ap.add_argument(
        "--trt_max_workspace_size",
        type=int,
        default=2,
        help="maximum workspace size of TensorRT engine",
    )

    return ap
