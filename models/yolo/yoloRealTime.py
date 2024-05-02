import cv2
from ultralytics import YOLO
from utils.readLicensePlate import readLicensePlate
from utils.utils import visualize
from sort.sort import *

# load models
license_plate_detector = YOLO('models/yolo/best.pt')
mot_tracker = Sort()
def yoloRealTimeDetect() :
    global license_plateScore
    pointx1 = 0.0
    pointx2 = 0.0
    pointy1 = 0.0
    pointy2 = 0.0
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    threshold = 0.8
    carPlate_dict = {}
    while True:
        ret, img= cap.read()
        results = license_plate_detector(img,stream=True)
        detections_ = []
        # coordinates
        for r in results:
            boxes = r.boxes
            for boxA in boxes:
                if (boxA.conf[0] > 0.60):
                    boxAx1, boxAy1, boxAx2, boxAy2 = boxA.xyxy[0].tolist()
                    license_plateScore = boxA.conf[0].tolist()
                    detections_.append([boxAx1, boxAy1, boxAx2, boxAy2, license_plateScore])

            print(len(detections_))
            detections_np = np.array(detections_).reshape(-1, 5)
            detections_bboxes = detections_np[:, :4]
            track_ids = mot_tracker.update(np.asarray(detections_bboxes))
            print(len(track_ids))
            if track_ids.shape[0] > 0 and track_ids.shape[1] == 5:
                for track_id in track_ids:
                        pointx1, pointy1, pointx2, pointy2, carPlate_id = track_id
                        imgCopy = img.copy()
                        license_plate_result = readLicensePlate(imgCopy,pointx1,pointy1,pointx2,pointy2)
                        if carPlate_id in carPlate_dict:
                            print(carPlate_id)
                            print(carPlate_dict[carPlate_id])

                            dict_license_plate_data = carPlate_dict[carPlate_id][0]
                            print(dict_license_plate_data)
                            dict_license_plate, dict_license_plate_score = dict_license_plate_data
                            visualize(img, license_plateScore, dict_license_plate, pointx1, pointy1, pointx2,
                                      pointy2)
                            # if dict_license_plate_score < license_plate_result[1]:
                            #     carPlate_dict.update({carPlate_id: [(license_plate_result[0],license_plate_result[1])]})
                            #     visualize(img, license_plateScore, dict_license_plate, pointx1, pointy1, pointx2, pointy2)
                            # else:

                        else:
                            carPlate_dict[carPlate_id] = [(license_plate_result[0],license_plate_result[1])]
                            visualize(img, license_plateScore, license_plate_result[0], pointx1, pointy1, pointx2, pointy2)

        print(carPlate_dict)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

yoloRealTimeDetect()