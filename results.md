Jetson Xavier NX:

ONNX Model [TensorRT engine]
10 runs FPS: [9.31, 9.25, 9.19, 9.3, 9.36, 9.38, 9.41, 9.37, 9.38, 9.41]

Pytorch Model
10 runs FPS: [11.3, 11.73, 11.69, 11.79, 11.83, 11.77, 11.73, 11.78, 11.8, 11.7]

Jetson Orin:

Pytorch Model [using CUDA]
1 run FPS: 18.88
1 run with writer FPS: 9.91
10 runs FPS: [18.82, 18.91, 18.87, 18.84, 18.91, 18.78, 18.81, 18.73, 18.88, 18.86]

Pytorch Model [without CUDA]
1 run FPS: 6.67

ONNX Model [TensorRT engine]
10 runs FPS: [16.82, 16.82, 16.84, 16.78, 16.81, 16.81, 16.85, 16.79, 16.8, 16.78]
each run cost ~89s

31/03/2024 - Jetson Xavier NX
Pytorch+CUDA
FPS: [12.43, 12.98, 13.04, 12.96, 12.95, 12.96, 12.97, 12.96, 13.03, 13.02]

ONNX+CUDA
FPS: [10.93, 12.92, 12.89, 12.94, 12.87, 12.91, 12.89, 12.92, 12.87, 12.92]
with IOBinding
FPS: [10.65, 12.9, 12.95, 12.95, 12.97, 12.96, 12.96, 12.98, 12.97, 12.98]

ONNX+TensorRT trt_max_workspace_size:1GB(default)
FPS: [12.86, 13.01, 13.04, 12.96, 11.68, 11.79, 11.98, 11.89, 12.7, 12.78]

ONNX+TensorRT trt_max_workspace_size:2GB
FPS: [13.15, 13.29, 12.66, 13.39, 13.41, 13.44, 13.41, 13.42, 13.36, 13.43]
FPS: [12.8, 12.9, 13.03, 12.9, 12.94, 12.91, 12.98, 12.91, 12.97, 12.94]
run on cached engine
FPS: [13.59, 13.79, 13.81, 13.5, 13.48]
FPS: [13.37, 13.44, 13.52, 13.39, 13.58, 13.53, 13.49, 13.53, 13.51, 13.43]
with IOBinding
FPS: [13.3, 13.37, 13.31, 13.36, 13.29, 13.29, 13.35, 13.33, 13.33, 13.39]

ONNX+TensorRT trt_max_workspace_size:3GB
FPS: [12.75, 12.5, 12.29, 12.46, 12.5, 12.66, 12.72, 12.76, 12.72, 12.77]

ONNX+TensorRT trt_max_workspace_size:4GB
FPS: [12.84, 12.95, 12.95, 13.0, 12.96, 13.01, 13.0, 13.0, 12.98, 13.01]
FPS: [12.84, 12.98, 12.95, 13.07, 12.98, 12.95, 13.03, 13.03, 13.02, 13.03, 13.04, 13.05, 13.1, 13.09, 13.06]
run on cached engine
FPS: [13.45, 13.6, 13.66, 13.72, 13.66, 13.51, 13.52, 13.51, 13.5, 13.4]

Observations:

-   While using jtop, fps drop 1 unit
-   Increase gpu_mem_limit for ONNX+CUDA has no effect

Jetson Xavier NX

-   cuda 11.4
-   jetpack 5.1.1

RTX4090

-   cuda 12.3

TODO:

1. Implementation

    - TensorRT Engine
    - Quantization

2. Accuracy Review
3. Experiments
    - PyTorch+CPU
        - Xavier NX
        - Orin
        - RTX4090   > FPS: [40.13, 40.19, 39.78, 40.33, 39.73, 40.68, 41.34, 40.87, 41.39, 40.82]
    - PyTorch+CUDA
        - Xavier NX > FPS: [12.43, 12.98, 13.04, 12.96, 12.95, 12.96, 12.97, 12.96, 13.03, 13.02]
        - Orin
        - RTX4090   > FPS: [90.27, 91.81, 92.66, 91.16, 92.22, 91.84, 91.08, 91.17, 92.3, 91.17]
    - ONNX+CPU
        - Xavier NX
        - Orin
        - RTX4090
    - ONNX+CUDA
        - Xavier NX > FPS: [10.82, 12.97, 12.97, 12.95, 12.95, 12.95, 13.02, 13.03, 13.01, 12.97]
        - Orin
        - RTX4090
    - ONNX+CUDA (IOBinding)
        - Xavier NX > FPS: [10.65, 12.9, 12.95, 12.95, 12.97, 12.96, 12.96, 12.98, 12.97, 12.98]
        - Orin
        - RTX4090
    - ONNX+TensorRT (trt_max_workspace_size = 1GB)
        - Xavier NX > FPS: [13.42, 13.51, 13.49, 13.53, 13.6, 13.59, 13.52, 13.45, 13.54, 13.51]
        - Orin
        - RTX4090
    - ONNX+TensorRT (trt_max_workspace_size = 2GB)
        - Xavier NX > FPS: [13.37, 13.44, 13.52, 13.39, 13.58, 13.53, 13.49, 13.53, 13.51, 13.43]
        - Orin
        - RTX4090
    - ONNX+TensorRT (trt_max_workspace_size = 4GB)
        - Xavier NX > FPS: [13.45, 13.6, 13.66, 13.72, 13.66, 13.51, 13.52, 13.51, 13.5, 13.4]
        - Orin
        - RTX4090
    - ONNX+TensorRT (trt_max_workspace_size = 2GB & IOBinding)
        - Xavier NX > FPS: [13.3, 13.37, 13.31, 13.36, 13.29, 13.29, 13.35, 13.33, 13.33, 13.39]
        - Orin
        - RTX4090
    - TensorRT
        - Xavier NX
        - Orin
        - RTX4090
