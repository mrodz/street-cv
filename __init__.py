import cv2
import numpy as np
import supervision as sv

from collections import defaultdict
from torch import Tensor
from typing import Optional
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

class VideoStream:  
    def __init__(self, source: cv2.VideoCapture) -> None:
        self.source = source
        self.track_history = defaultdict(lambda: [])
        self.painter = sv.BoxAnnotator(
            thickness=WINDOW_LINE_THICKNESS,
            text_thickness=WINDOW_TEXT_THICKNESS,
            text_scale=WINDOW_TEXT_SCALE
        )
        
    def __iter__(self):
        return self
    
    def update_tracking(self, this_frame_tracks: Tensor, detected_idx: int, pos: tuple[float, float]):
        # copy to new variable to avoid side effects
        track_ids_list = this_frame_tracks.cpu().int().tolist()
        track_id = track_ids_list[detected_idx]
                    
                    
        if self.track_history.get(track_id) is None:
            self.track_history[track_id] = [pos]
        else:
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
                
            self.track_history[track_id].append(pos)                    

        return self.track_history.get(track_id)
    
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

                    track = self.update_tracking(track_ids, idx, center_of_box)
                    
                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 0xff, 0), thickness=15)

            
                detections.xyxy[idx] = (x1, y1, x2, y2)
            
                labels.append(f"{MODEL.model.names[id]} {cf:0.2f}")
            
            
        frame = self.painter.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        
        return frame


def main():
    capture = None
    
    try:
        print("Opening camera port", CAMERA_INPUT_PORT, "...")
        capture = cv2.VideoCapture(CAMERA_INPUT_PORT, cv2.CAP_DSHOW)
        print("Camera opened")
    
        print("Setting dimensions...")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_DIMS[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_DIMS[1])
        capture.set(cv2.CAP_PROP_FPS, 60)
        print("Done")

    
        stream = VideoStream(capture)

        for frame in stream:
            if frame is None:
                # cv2 camera IN stream is lagging behind: skip this render
                continue

            cv2.imshow("YOLOv8", frame)

            if (cv2.waitKey(30) == 27): # break with escape key
                break

    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()