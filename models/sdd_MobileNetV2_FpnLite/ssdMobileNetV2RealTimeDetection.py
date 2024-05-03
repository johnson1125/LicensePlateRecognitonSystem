import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
from utils.utils import get_car
from utils.readLicensePlate import readLicensePlate
from utils.utils import visualize
from sort.sort import *
from ultralytics import YOLO

# setup
coco_model = YOLO('models/yolov8n.pt')
modelpath = 'models/sdd_MobileNetV2_FpnLite/detect.tflite'
min_conf = 0.50
mot_tracker = Sort()
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

def ssdRealTimeModelDetect():

    #Configure Camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    vehicles = [2, 3, 5, 7]
    #Declare CarPlate dictionary
    carPlate_dict = {}
    while (True):
        ret, frame = cap.read()

        #Yolo COCO pretrained model detection
        detections = coco_model(frame)[0]
        detections_ = []

        #Yolo COCO pretrained model result
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        detections_np = np.array(detections_).reshape(-1, 5)
        detections_bboxes = detections_np[:, :4]
        track_ids = mot_tracker.update(np.asarray(detections_bboxes))

        #Preprocessing frame for ssdMobileNetV2 detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        detections_ = []
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

        license_plate_result = None

        # ssdMobilenetV2 result
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                license_plateScore = scores[i]
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                license_plate = (xmin,ymin,xmax,ymax,license_plateScore,"")

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                frameCopy = frame.copy()

                # run license plate recognition
                license_plate_result = readLicensePlate(frameCopy, xmin,ymin,xmax,ymax)

                # check the car plate for current car is recorded in the car plate dictionary
                if car_id in carPlate_dict:
                    # retrieve car plate of current car from the car plate dictionary
                    dict_license_plate_data = carPlate_dict[car_id][0]
                    dict_license_plate, dict_license_plate_score = dict_license_plate_data

                    # compare the precision of latest license plate recognition with the car plate recorded in the car plate dictionary
                    if dict_license_plate_score < license_plate_result[1]:
                        # update the car plate record to better precision car plate record
                        carPlate_dict.update({car_id: [(license_plate_result[0], license_plate_result[1])]})
                        # annotate the result to the frame
                        visualize(frame, score, license_plate_result[0], xmin,ymin,xmax,ymax)
                    else:
                        # annotate the result to the frame
                        visualize(frame, score, dict_license_plate, xmin,ymin,xmax,ymax)

                else:
                    # save license plate recognition to car plate dictionary
                    carPlate_dict[car_id] = [(license_plate_result[0], license_plate_result[1])]
                    # annotate the result to the frame
                    visualize(frame, score, license_plate_result[0], xmin,ymin,xmax,ymax)

                with open("resources/registered_car_plate.txt", 'r') as file:
                    # Read all lines from the file
                    lines = file.readlines()
                    # Strip newline characters from each line and compare with the string value
                    for line in lines:
                        if line.strip() == license_plate_result[0]:
                            cap.release()
                            cv2.destroyAllWindows()
                            return license_plate_result[0]

        cv2.imshow('RealTime license Plate System', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()