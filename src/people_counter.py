import argparser
import rapid_engines
import RAPiD.utils.visualization as rapid_visual

import time

import cv2
import dlib
from imutils.video import VideoStream, FPS


def parse_detection(detection):
    if len(detection) == 6:
        cX, cY, width, height, angle, conf = detection
    else:
        cX, cY, width, height, angle = detection[:5]
        conf = -1

    return cX, cY, width, height, angle, conf


def dlib_tracker_init(detection, frame_rgb):
    cX, cY, width, height, _, _ = parse_detection(detection)

    # Find topleft and bottomright corners of the bounding box
    startX, startY = int(cX - width / 2), int(cY - height / 2)
    endX, endY = startX + width, startY + height

    # Initialise dlib correlation tracker
    tracker = dlib.correlation_tracker()
    # Create a dlib rectangle object
    rect = dlib.rectangle(startX, startY, endX, endY)
    # Start tracking the detection
    tracker.start_track(frame_rgb, rect)

    return tracker


def draw_detection(frame, detection):
    cX, cY, width, height, angle, conf = parse_detection(detection)

    rapid_visual.draw_xywha(frame, cX, cY, width, height, angle)
    cv2.putText(
        frame,
        f"{conf:.2f}",
        (int(cX), int(cY)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_tracking_object(frame, tracker):
    pos = tracker.get_postion()
    # unpack position object
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 5)


def main():
    recursion_count = 0
    results = []

    # Construct and parse command-line arguments
    ap = argparser.create_parser()
    args = ap.parse_args()

    # Initialise the inference engine for RAPiD
    engine = rapid_engines.create_engine(
        model_path=args.weights,
        engine_type=args.engine_type,
        execution_provider=args.execution_provider,
        input_size=args.framesize,
        conf_thres=args.confidence,
    )

    for recursion_count in range(args.recursion):
        video_writer = None
        frame_count = 0
        trackers = []

        # Initialise a video stream from webcam if no input path is provided
        if args.input is None:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
        # Otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(args.input)

        if vs.isOpened():
            frame_width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Initialise a video writer if output video is requested
        if args.output is not None:
            # Only process the video once
            args.recursion = 1
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(
                args.output, fourcc, 30, (frame_width, frame_height), True
            )

        draw_frame = True if video_writer is not None or args.display else False

        # Start the FPS throughput estimator
        fps = FPS().start()

        # Loop over each frame of video file or video stream (webcam)
        while True:
            # Read the next frame
            frame = vs.read()
            frame = frame[1] if args.input is not None else frame

            # Base case: if it's the end of the video
            if args.input is not None and frame is None:
                break

            # dlib and PIL Image conversion need RGB ordering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run RAPiD every N frames
            if frame_count % args.skip_frames == 0:
                # Reset trackers
                trackers.clear()

                # Convert and resize frame
                frame_input = engine.preprocess_frame(frame_rgb)

                detections = engine.infer(frame_input)

                # Loop over each detection
                for dt in detections:
                    # Initialise dlib tracker to track detection
                    tracker = dlib_tracker_init(dt, frame_rgb)
                    trackers.append(tracker)

                    if draw_frame:
                        draw_detection(frame, dt)

            # Run dlib correlation tracker
            else:
                for tracker in trackers:
                    # Update the object tracker
                    tracker.update(frame_rgb)

                    if draw_frame:
                        draw_tracking_object(frame, tracker)

            if draw_frame:
                people_count = len(trackers)
                cv2.putText(
                    frame,
                    f"Count: {people_count}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

            if video_writer is not None:
                video_writer.write(frame)

            if args.display:
                cv2.imshow("RAPiD People Counter", frame)
                key = cv2.waitKey(1) & 0xFF
                # Stop processing if 'q' key is pressed
                if key == ord("q"):
                    break

            # Update the FPS timer
            fps.update()
            frame_count += 1

            # Inform status every 300 frames
            if frame_count % 300 == 0:
                print(f"[INFO] {frame_count} frames are processed")

        # Stop the FPS timer
        fps.stop()
        # Store result of each run
        results.append(round(fps.fps(), 2))
        # Display result of each run
        print(
            "[INFO] [{}/{}] elapsed time: {:.2f}".format(
                recursion_count + 1, args.recursion, fps.elapsed()
            )
        )
        print(
            "[INFO] [{}/{}] approx. FPS: {:.2f}".format(
                recursion_count + 1, args.recursion, fps.fps()
            )
        )
        print(
            "[INFO] [{}/{}] processed frames: {}".format(
                recursion_count + 1, args.recursion, frame_count
            )
        )

        # Close the video stream or release the video file pointer
        if args.input is None:
            vs.stop()
        else:
            vs.release()

        # Close the display window of processed frames
        if args.display:
            cv2.destroyAllWindows()

        # Release the video writer
        if video_writer is not None:
            video_writer.release()

    print("[INFO] FPS: ", results)


if __name__ == "__main__":
    main()
