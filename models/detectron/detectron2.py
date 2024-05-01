from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2

# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = 'models/detectron/lastest_detectron2.pth'
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# Create predictor instance
predictor = DefaultPredictor(cfg)

# Load image
image = cv2.imread("./photo6.jpg")

# Perform prediction
outputs = predictor(image)

threshold = 0.8

# Display predictions
preds = outputs["instances"].pred_classes.tolist()
scores = outputs["instances"].scores.tolist()
bboxes = outputs["instances"].pred_boxes

for j, bbox in enumerate(bboxes):
    bbox = bbox.tolist()

    score = scores[j]
    pred = preds[j]

    if score > threshold:
        x1, y1, x2, y2 = [int(i) for i in bbox]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # Add prediction score as text
        text = f"Class {pred} ({score:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
