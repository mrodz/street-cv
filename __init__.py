import cv2
import numpy as np
import supervision as sv
import time

from collections import defaultdict
from cv2.typing import MatLike
from torch import Tensor
from typing import Optional, Callable
from ultralytics import YOLO

"""
CONFIG DETAILS
"""
PYTORCH_MODEL_PATH = "/data/yolov8n.pt"
WINDOW_LINE_THICKNESS = 2
WINDOW_TEXT_THICKNESS = 2
WINDOW_TEXT_SCALE = 1
CAMERA_INPUT_PORT = 0
WINDOW_DIMS = (1200, 800)
SCALE_FACTOR = 0.25

CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
PERSON = 0

print("Loading model...")
MODEL = YOLO(PYTORCH_MODEL_PATH)
print("Loaded.")


def scale_xy_down(xy: tuple[int, int]) -> tuple[int, int]:
    return (int(xy[0] * SCALE_FACTOR), int(xy[1] * SCALE_FACTOR))
   
    
def scale_xy_up(xy: tuple[int, int]) -> tuple[int, int]:
    return (xy[0] // SCALE_FACTOR, xy[1] // SCALE_FACTOR)


# Numpy doesn't like sets, so we have to use a plain array
VEHICLE_IDS = [CAR, MOTORCYCLE, BUS, TRUCK, PERSON]


class TrackedObject:
    
    __slots__ = 'positions', 'at_intersection'
    
    def __init__(self) -> None:
        self.positions = []
        self.at_intersection = None
        
    def add(self, pos: tuple[float, float]):
        self.positions.append(pos)

        if len(self.positions) > 30:
            self.positions.pop(0)
            
    def draw_to_frame(self, frame: MatLike):
        points = np.hstack(self.positions).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(0, 0xff, 0), thickness=15)


class TrackedObjects:
    
    __slots__ = 'objects'
    
    def __init__(self) -> None:
        self.objects = defaultdict(lambda: TrackedObject())
    
    def update_tracking(self, tracking_id: int, pos: tuple[float, float]) -> TrackedObject:
        track: TrackedObject = self.objects[tracking_id]
        track.add(pos)
        return track
    
    def __contains__(self, item: int):
        return item in self.objects


class VideoStream:
    
    __slots__ = 'source', 'track_history', 'painter'
    
    def __init__(self, source: cv2.VideoCapture) -> None:
        self.source = source
        self.track_history = TrackedObjects()
        self.painter = sv.BoxAnnotator(
            thickness=WINDOW_LINE_THICKNESS,
            text_thickness=WINDOW_TEXT_THICKNESS,
            text_scale=WINDOW_TEXT_SCALE
        )
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Optional[np.ndarray]:
        ok, frame = self.source.read()

        if not ok:
            print("WARNING: OpenCV read returned false on input read")
            return None

        height, width, _ = frame.shape
        
        downscaled = scale_xy_down((width, height))
        
        resized = cv2.resize(frame, downscaled, interpolation=cv2.INTER_AREA)
        
        result = MODEL.track(resized, agnostic_nms=True, verbose=False, persist=True)[0]

        track_ids: Optional[Tensor] = result.boxes.id
        
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, VEHICLE_IDS) & (detections.confidence > .5)]
        
        labels = list()
        
        for idx in range(len(detections.xyxy)):
            class_id = detections[idx].class_id
            confidence = detections[idx].confidence
            if (id := class_id) is not None and (cf := confidence) is not None:    
                
                id = id[0]
                cf = cf[0]
            
                x1, y1, x2, y2 = detections.xyxy[idx]
            
                x1, y1 = scale_xy_up((x1, y1))
                x2, y2 = scale_xy_up((x2, y2))
            
                if track_ids is not None:
                    center_of_box = (float((x2 + x1) / 2), float((y2 + y1) / 2))

                    track_ids_list = track_ids.cpu().int().tolist()
                    track_id = track_ids_list[idx]

                    track: TrackedObject = self.track_history.update_tracking(track_id, center_of_box)
                    
                    # Draw the tracking lines
                    track.draw_to_frame(frame)

            
                detections.xyxy[idx] = (x1, y1, x2, y2)
            
                labels.append(f"{MODEL.model.names[id]} {cf:0.2f}")
            
            
        frame = self.painter.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        
        return frame


def format_cv2_input(getter: Callable[[], cv2.VideoCapture]) -> Callable[[], cv2.VideoCapture]:
    def inner():
        print("Getting input source...")
        capture = getter()
        print("Done:", capture)
        
        print("Waiting for device:")
        
        i = 0
        while not capture.isOpened():
            dots = (i % 3 + 1)
            print("." * dots, " " * (3 - dots), " (", i, "s elapsed)", sep="", end="\r")
            
            time.sleep(1)

            i += 1
            
        print("Ready")
    
        print("Setting dimensions...")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_DIMS[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_DIMS[1])
        capture.set(cv2.CAP_PROP_FPS, 30)
        print("Done")
        return capture
        
    return inner


@format_cv2_input
def live_capture() -> cv2.VideoCapture:  
    return cv2.VideoCapture(CAMERA_INPUT_PORT, cv2.CAP_DSHOW)


@format_cv2_input
def video_capture() -> cv2.VideoCapture:
    return cv2.VideoCapture("./data/camera_data_from_yt_0.avi")


def main():
    capture = None
    
    try:
        capture = live_capture()
        
        stream = VideoStream(capture)

        # used to record the time when we processed last frame 
        prev_frame_time = 0
  
        # used to record the time at which we processed current frame 
        new_frame_time = 0

        for frame in stream:
            # time when we finish processing for this frame 
            new_frame_time = time.time() 
  
            fps = 1/(new_frame_time-prev_frame_time) 
            prev_frame_time = new_frame_time 

            if frame is None:
                # cv2 camera IN stream is lagging behind: skip this render
                continue

            cv2.putText(frame, f"{fps:.2f} fps", (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA) 

            cv2.imshow("YOLOv8", frame)

            if (cv2.waitKey(30) == 27): # break with escape key
                break

    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()