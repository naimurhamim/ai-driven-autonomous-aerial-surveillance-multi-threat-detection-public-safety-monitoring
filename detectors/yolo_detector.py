from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str, conf: float = 0.4):
        self.model = YOLO(model_path)  # e.g. "yolo11n.pt" or your trained weights
        self.conf = conf

    def infer(self, frame):
        # returns Ultralytics Results list
        return self.model.predict(source=frame, conf=self.conf, verbose=False)