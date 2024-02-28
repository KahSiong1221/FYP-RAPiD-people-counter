from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image
import imutils
import argparse
import time
import cv2
import torch
import numpy as np

from PeopleTracker.peopletracker import PeopleTracker
import dlib

import onnx, onnxruntime

from RAPiD.api import Detector
from RAPiD.utils import utils, visualization


def post_processing(dts, pad_info, conf_thres=0.3, nms_thres=0.3):
	assert isinstance(dts, torch.Tensor)
	dts = dts[dts[:,5] >= conf_thres].cpu()
	dts = utils.nms(dts, is_degree=True, nms_thres=nms_thres)
	dts = utils.detection2original(dts, pad_info.squeeze())
	return dts


# Construct and parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-w",
    "--weights",
    required=True,
    type=str,
    help="path to required pre-trained network weights",
)
ap.add_argument(
    "--use-cuda",
    action="store_true",
    default=False,
    help="use Nvidia CUDA GPU",
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
    "--framesize",
    type=int,
    default=1024,
    help="frame size of input video/webcam",
)
ap.add_argument(
    "--display",
    action="store_true",
    default=False,
    help="display processed frames on screen",
)
ap.add_argument(
    "--repeat",
    type=int,
    default=1,
    help="# of times to run the counter on the same input video",
)
args = vars(ap.parse_args())

'''
# Initialize the RAPiD object detector
rapid_detector = Detector(
    model_name="rapid",
    weights_path=args["weights"],
    use_cuda=args["use_cuda"],
    input_size=args["framesize"],
    conf_thres=args["confidence"],
)
'''

ort_session = onnxruntime.InferenceSession(
    args["weights"], 
    providers=[
    	'TensorrtExecutionProvider',
    	'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ])
    
print("[INFO] ONNX runtime session providers: {}".format(ort_session.get_providers()))

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
io_binding = ort_session.io_binding()

repeat_count = 0
fps_results = []

for repeat_count in range(args["repeat"]):
	# Initialize a video stream from webcam if no input video path
	if not args.get("input", False):
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)
	# Otherwise, grab a reference to the video file
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(args["input"])

	# Initialize the video writer
	writer = None

	# Initialize the frame dimensions
	W = None
	H = None

	# Instantiate the people tracker and the dlib correliation trackers
	pt = PeopleTracker(maxDisappeared=40, maxDistance=50)
	trackers = []

	peopleCount = 0
	totalFrames = 0

	# Start the FPS throughput estimator
	fps = FPS().start()

	# Until the end of video file or video stream (webcam)
	while True:
		# read the next frame from video file or webcam
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# if its the end of the video
		if args["input"] is not None and frame is None:
			break

		frame = imutils.resize(frame, width=args["framesize"])

		# convert the frame from openCV format to PIL format for RAPiD
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pil_frame = Image.fromarray(rgb)

		img_resized, _, pad_info = utils.rect_to_square(pil_frame, None, args["framesize"])
		im_numpy = np.expand_dims(np.array(img_resized), 0).transpose(0,3,1,2).astype(np.float32) / 255.0


		# get the frame size (width, height)
		if W is None or H is None:
			(W, H) = pil_frame.size

		# if output video is required, initialize the writer
		if args["output"] is not None and writer is None:
			args["repeat"] = 1
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

		# initialize the list of bounding rectangles returned by object
		# detector or the correlation trackers
		rects = []

		# run RAPiD object detection every N frames, N = num of skip frames
		if totalFrames % args["skip_frames"] == 0:
			trackers = []

			# forward the frame through the neural network for person detection
			# returns a list of [cx,cy,w,h,a,conf]
			### detections = rapid_detector.detect_one(pil_img=pil_frame)

            im_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(im_numpy, 'cuda', 0)
            
            io_binding.bind_input(name='input', device_type=im_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=im_ortvalue.shape(), buffer_ptr=im_ortvalue.data_ptr())
            io_binding.bind_output('output')
			# ort_outputs = ort_session.run([], {input_name: im_numpy})
            ort_session.run_with_iobinding(io_binding)
			detections = io_binding.copy_outputs_to_cpu()[0].squeeze(0)

			# post-processing
			detections = torch.from_numpy(detections)
			detections = post_processing(detections, pad_info, conf_thres=0.3, nms_thres=0.3)

			for i in range(len(detections)):
				# parse the detection
				if len(detections[i]) == 6:
					cX, cY, width, height, angle, conf = detections[i]
				else:
					cX, cY, width, height, angle = detections[i][:5]
					conf = -1

				# draw rotated identified bounding rectangles
				visualization.draw_xywha(frame, cX, cY, width, height, angle)

				startX = int(cX - width / 2)
				startY = int(cY - height / 2)
				endX = startX + width
				endY = startY + height

				# instantiate dlib correlation tracker
				tracker = dlib.correlation_tracker()
				# create a dlib rectangle object by topleft corner and bottom right corner
				rect = dlib.rectangle(startX, startY, endX, endY)
				# start tracking the identified person
				tracker.start_track(rgb, rect)
				trackers.append(tracker)

		# run the correlation tracker instead of RAPiD detector
		else:
			for tracker in trackers:
				# update the tracker and get the updated position
				tracker.update(rgb)
				pos = tracker.get_position()
				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())
				# add bounding rectangle to the list
				rects.append((startX, startY, endX, endY))

		# update centroids and bounding rectangles in person tracker
		objects, boxes = pt.update(rects)

		# for each tracked object
		for (objectID, centroid), (_, box) in zip(objects.items(), boxes.items()):
		    # display the centroid, bounding rectangles and ID
		    # of the object on the output frame
		    text = "ID {}".format(objectID)
		    cv2.putText(
		        frame,
		        text,
		        (centroid[0] - 10, centroid[1] - 10),
		        cv2.FONT_HERSHEY_SIMPLEX,
		        0.5,
		        (0, 255, 0),
		        2,
		    )
		    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)

		# calculate and show the number of people count on the output frame
		peopleCount = len(objects)
		cv2.putText(
		    frame,
		    "Count: {}".format(peopleCount),
		    (20, 70),
		    cv2.FONT_HERSHEY_SIMPLEX,
		    2.0,
		    (0, 255, 0),
		    3,
		)

		# write the output frame to disk if the write is on
		if writer is not None:
		    writer.write(frame)

		# show the output frame
		if args["display"]:
		    cv2.imshow("Frame", frame)
		    key = cv2.waitKey(1) & 0xFF
		    # stop processing if the `q` key was pressed
		    if key == ord("q"):
		        break

		# update the FPS counter
		totalFrames += 1
		fps.update()
		
		if totalFrames % 200 == 0:
			print("[INFO] {} frames are processed".format(totalFrames))

	# Stop FPS timer and display FPS info
	fps.stop()
	fps_results.append(round(fps.fps(), 2))
	print("[INFO] [{}/{}] elapsed time: {:.2f}".format(repeat_count+1, args["repeat"], fps.elapsed()))
	print("[INFO] [{}/{}] approx. FPS: {:.2f}".format(repeat_count+1, args["repeat"], fps.fps()))
	print("[INFO] [{}/{}] processed frames: {}".format(repeat_count+1, args["repeat"], totalFrames))
	
	# Stop the video stream if no input video
	if not args.get("input", False):
		vs.stop()
	# Otherwise, release the video file pointer
	else:
		vs.release()

	cv2.destroyAllWindows()

# Release the video writer
if writer is not None:
    writer.release()

print("[INFO] FPS: ", fps_results)

