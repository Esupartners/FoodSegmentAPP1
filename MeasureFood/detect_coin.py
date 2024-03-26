from ultralytics import YOLO


MODEL_PATH = r'coin_detector.pt'

PREDICT_ARGS = {
    # Detection Settings
    'conf': 0.5,  # object confidence threshold for detection
    'iou': 0.7,  # intersection over union (IoU) threshold for NMS
    'imgsz': 640,  # image size as scalar or (h, w) list, i.e. (640, 480)
    'half': False,  # use half precision (FP16)
    'device': None,  # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    'max_det': 50,  # maximum number of detections per image
    }

def load_model(model_path='yolov8.pt'):

    # Load a model
    model = YOLO(model=model_path,task='detect')

    return model


def detect_coin(image_path=None,model_path=MODEL_PATH):

    model = load_model(model_path)

    # Run inference
    results = model.predict(source=image_path,**PREDICT_ARGS)[0]

    bounding_boxes = results.boxes.xywh
    scores = results.boxes.conf
    classes = results.boxes.cls

    return classes.cpu().numpy(),bounding_boxes.cpu().numpy(), scores.cpu().numpy()