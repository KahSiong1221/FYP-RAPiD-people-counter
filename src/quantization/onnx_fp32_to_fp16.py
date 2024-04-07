import onnx
from onnxconverter_common import float16
import argparse

ap = argparse.ArgumentParser()

ap.add_argument(
    "-i",
    "--input",
    type=str,
    help="path to input ONNX model (FP32) file",
)

ap.add_argument(
    "-o",
    "--output",
    type=str,
    help="path to output ONNX model (FP16) file",
)

args = ap.parse_args()

model = onnx.load(args.input)

model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

onnx.save(model_fp16, args.output)
