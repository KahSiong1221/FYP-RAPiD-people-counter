Jetson Xavier NX:

ONNX Model [TensorRT engine]
10 runs FPS:  [9.31, 9.25, 9.19, 9.3, 9.36, 9.38, 9.41, 9.37, 9.38, 9.41]

Pytorch Model
10 runs FPS:  [11.3, 11.73, 11.69, 11.79, 11.83, 11.77, 11.73, 11.78, 11.8, 11.7]

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
