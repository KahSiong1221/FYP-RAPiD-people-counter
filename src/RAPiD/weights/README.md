# Weights
Put the `.ckpt` files in this folder.

PyTorch pre-trained RAPiD model
https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB1024_Mar11_4000.ckpt

Run /RAPiD/export_onnx.py to convert the PyTorch model to ONNX model (input shape 1x4x1024x1024).

Run /RAPiD/export_onnx608.py to convert the PyTorch model to ONNX model (input shape 1x3x608x608).

Run /src/quantization/onnx_fp32_to_fp16.py to perform FP16 quantization on RAPiD onnx model.

Run /src/quantization/onnx_fp32_to_int8.py to generate calibration table for building the INT8 TensorRT Engine (calibration dataset is required, I used a subset of CEPDOF dataset).