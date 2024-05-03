import cv2
from models.faster_rcnn.detectron2.config import get_cfg
from models.faster_rcnn.detectron2.engine import DefaultPredictor
from models.faster_rcnn.detectron2 import model_zoo
from utils.readLicensePlate import readLicensePlate
from utils.utils import visualize
from PIL import Image, ImageTk

# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = 'models/faster_rcnn/faster_rcnn.pth'
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

def fasterRcnnRealTimeDetect():
    # Create predictor instance
    predictor = DefaultPredictor(cfg)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, img = cap.read()

        # Perform prediction
        outputs = predictor(img)

        threshold = 0.8

        # Display predictions
        scores = outputs["instances"].scores.tolist()
        bboxes = outputs["instances"].pred_boxes

        for j, bbox in enumerate(bboxes):
            bbox = bbox.tolist()
            score = scores[j]

            if score > threshold:
                x1, y1, x2, y2 = [int(i) for i in bbox]

                imgcpy = img.copy()
                license_plate = readLicensePlate(imgcpy, x1, y1, x2, y2)

                visualize(img, score, license_plate, x1, y1, x2, y2)

                with open("resources/registered_car_plate.txt", 'r') as file:
                    # Read all lines from the file
                    lines = file.readlines()
                    # Strip newline characters from each line and compare with the string value
                    for line in lines:
                        if line.strip() == license_plate[0]:
                            cap.release()
                            cv2.destroyAllWindows()
                            return license_plate[0]

        cv2.imshow('RealTime license Plate System', img)

        if cv2.waitKey(1) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


