import cv2

def visualize (img, license_plateScore, license_plate, x1,y1,x2,y2):
    label = "%d%% %s" % (int(license_plateScore * 100),license_plate)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)