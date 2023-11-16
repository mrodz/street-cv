from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

PYTORCH_MODEL_PATH = "/data/yolov8n.pt"
WINDOW_LINE_THICKNESS = 2
WINDOW_TEXT_THICKNESS = 2
WINDOW_TEXT_SCALE = 1
CAMERA_INPUT_PORT = 0
WINDOW_DIMS = (1200, 800)
SCALE_FACTOR = 0.1

print("Loading model...")
MODEL = YOLO(PYTORCH_MODEL_PATH)
print("Loaded.")

class VideoStream:  
    def __init__(self, source: cv2.VideoCapture) -> None:
        self.source = source
        self.painter = sv.BoxAnnotator(
            thickness=WINDOW_LINE_THICKNESS,
            text_thickness=WINDOW_TEXT_THICKNESS,
            text_scale=WINDOW_TEXT_SCALE
        )
        
    def __iter__(self):
        return self
    
    @staticmethod
    def scale_xy_down(xy: tuple[int, int]) -> tuple[int, int]:
        return (int(xy[0] * SCALE_FACTOR), int(xy[1] * SCALE_FACTOR))
    
    @staticmethod
    def scale_xy_up(xy: tuple[int, int]) -> tuple[int, int]:
        return (xy[0] // SCALE_FACTOR, xy[1] // SCALE_FACTOR)
        
    def __next__(self) -> np.ndarray:
        _, frame = self.source.read()

        height, width, _ = frame.shape
        
        downscaled = VideoStream.scale_xy_down((width, height))
        
        resized = cv2.resize(frame, downscaled, interpolation=cv2.INTER_AREA)
        
        result = MODEL(resized, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        for idx in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[idx]
            
            x1, y1 = VideoStream.scale_xy_up((x1, y1))
            x2, y2 = VideoStream.scale_xy_up((x2, y2))
            
            detections.xyxy[idx] = (x1, y1, x2, y2)


        labels = [
            f"{MODEL.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

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
            cv2.imshow("YOLOv8", frame)

            if (cv2.waitKey(30) == 27): # break with escape key
                break

    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()