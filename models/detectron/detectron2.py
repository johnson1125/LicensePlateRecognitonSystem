from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2

# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = './lastest_detectron2.pth'
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

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Add prediction score as text
        text = f" {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('image', image)
cv2.waitKey(0)

# # Initialize webcam capture
# video_capture = cv2.VideoCapture(0)  # Use default webcam (change if needed)
#
# while True:
#     # Capture frame from webcam
#     ret, frame = video_capture.read()
#
#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect objects
#     outputs = predictor(frame)
#
#     # Set a confidence threshold for visualization
#     threshold = 0.9
#
#     # Display bounding boxes and labels
#     for j, bbox in enumerate(outputs["instances"].pred_boxes):
#         score = outputs["instances"].scores[j]
#         if score > threshold:
#             x1, y1, x2, y2 = [int(i) for i in bbox.tolist()]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             class_id = outputs["instances"].pred_classes[j]
#             cv2.putText(frame, f"Class {class_id} ({score:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Webcam Object Detection', frame)
#
#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release webcam and close windows
# video_capture.release()
# cv2.destroyAllWindows()

