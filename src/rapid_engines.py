from abc import ABC, abstractmethod

from RAPiD.api import Detector
from RAPiD.utils import utils as rapid_utils

import PIL
import onnxruntime
import numpy as np
import torch

MODEL_NAME = "rapid"


def create_engine(model_path, engine_type, execution_provider, input_size, conf_thres):
    if engine_type == "pytorch":
        return PyTorchEngine(model_path, execution_provider, input_size, conf_thres)
    elif engine_type == "onnx":
        return ONNXEngine(model_path, execution_provider, input_size, conf_thres)
    elif engine_type == "tensorrt":
        pass
    else:
        raise ValueError("[ERROR] invalid engine type: {}".format(engine_type))


class RAPiDEngine(ABC):
    def __init__(self, input_size, conf_thres):
        self.input_size = input_size
        self.conf_thres = conf_thres

    @abstractmethod
    def preprocess_frame(self, frame_rgb):
        pass

    @abstractmethod
    def infer(self, frame_input):
        pass


class PyTorchEngine(RAPiDEngine):
    """PyTorchEngine class for RAPiD object detector with PyTorch backend.

    Args:
        model_path (str): Path to the pre-trained PyTorch network weights
        execution_provider (str): Execution provider to use for inference ('cpu' or 'cuda')
        input_size (int): Input resolution
        conf_thres (float): Confidence threshold
    """

    def __init__(
        self,
        model_path: str,
        execution_provider: str,
        input_size: int,
        conf_thres: float,
    ):
        super().__init__(input_size, conf_thres)

        self.engine = Detector(
            model_name=MODEL_NAME,
            weights_path=model_path,
            use_cuda=True if execution_provider == "cuda" else False,
            input_size=input_size,
            conf_thres=conf_thres,
        )

    def preprocess_frame(self, frame_rgb):
        """Preprocesses an input frame for the RAPiD object detector.

        This method takes an RGB frame and only converts it into PIL Image format (expected by RAPiD API).
        Also the RAPiD API itself is likely responsible for resizing the frame before infering.

        Args:
            frame_rgb (numpy.ndarray): Input frame in RGB ordering
        """
        return PIL.Image.fromarray(frame_rgb)

    def infer(self, frame_input):
        return self.engine.detect_one(pil_img=frame_input)


class ONNXEngine(RAPiDEngine):
    def __init__(self, model_path, execution_provider, input_size, conf_thres):
        # self.execution_provider = execution_provider
        super().__init__(input_size, conf_thres)

        # Configure ONNX execution providers
        providers = ["CPUExecutionProvider"]

        if execution_provider == "cuda" or execution_provider == "tensorrt":
            providers.insert(0, "CUDAExecutionProvider")
        if execution_provider == "tensorrt":
            providers.insert(0, "TensorrtExecutionProvider")

        # Initialise RAPiD ONNX runtime session
        self.engine = onnxruntime.InferenceSession(model_path, providers=providers)

        # Store input & output name of ONNX model
        self.input_name = self.engine.get_inputs()[0].name
        self.output_name = self.engine.get_outputs()[0].name

    def preprocess_frame(self, frame_rgb):
        frame_pil = PIL.Image.fromarray(frame_rgb)

        frame_resized, _, self.pad_info = rapid_utils.rect_to_square(
            frame_pil, None, self.input_size
        )

        frame_input = (
            np.expand_dims(np.array(frame_resized), 0)
            .transpose(0, 3, 1, 2)
            .astype(np.float32)
            / 255.0
        )

        return frame_input

    def postprocess_detections(self, detections):
        detections = torch.from_numpy(detections)

        detections = detections[detections[:, 5] >= self.conf_thres].cpu()
        detections = rapid_utils.nms(
            detections, is_degree=True, img_size=self.input_size
        )
        detections = rapid_utils.detection2original(detections, self.pad_info.squeeze())

        return detections

    def infer(self, frame_input):
        onnxrt_outputs = self.engine.run([], {self.input_name: frame_input})
        detections = onnxrt_outputs[0].squeeze(0)

        # Post-processing
        return self.postprocess_detections(detections)


class TensorRTEngine(RAPiDEngine):
    def __init__(self):
        pass

    def preprocess_frame(self):
        pass