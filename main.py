import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from models.yolo.yoloPhoto import yoloDetectPhoto
from models.yolo.yoloCamera import yoloRealTimeDetect
from models.sdd_MobileNetV2_FpnLite.sddMobileNetV2Photo import ssdDetectPhoto
from models.sdd_MobileNetV2_FpnLite.sddMobileNetV2Camera import ssdRealTimeDetect

def yoloPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = yoloDetectPhoto(path)
        img = Image.open(detectionResult[1])
        img = img.resize((640, 640))
        pic = ImageTk.PhotoImage(img)

        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("1000x800")
        label.config(image=pic)
        label.image = pic
        print(detectionResult[0])

    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")

def ssdPhotoDetection():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        detectionResult = ssdDetectPhoto(path)
        img = Image.open(detectionResult[1])
        img = img.resize((640, 640))
        pic = ImageTk.PhotoImage(img)

        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("1000x800")
        label.config(image=pic)
        label.image = pic
        print(detectionResult[0])

    # if no file is selected, then we are displaying below message
    else:
        print("No file is chosen !! Please choose a file.")

def yoloRealTimeDetection():
    yoloRealTimeDetect()

def ssdRealTimeDetection():
    ssdRealTimeDetect()

if __name__ == "__main__":
    # defining tkinter object
    app = tk.Tk()

    # setting title and basic size to our App
    app.title("GeeksForGeeks Image Viewer")
    app.geometry("1000x800")

    # adding background image
    img = ImageTk.PhotoImage(file='resources/image/test/photo1.jpg')
    imgLabel = Label(app, image=img)
    imgLabel.place(x=0, y=0)

    # adding background color to our upload button
    app.option_add("*Label*Background", "white")
    app.option_add("*Button*Background", "lightgreen")

    label = tk.Label(app)
    label.pack(pady=10)

    # defining yolo detect Image buttom
    yoloDetectImageButton = tk.Button(app, text="Detect Image (Yolo)", command=yoloPhotoDetection)
    yoloDetectImageButton.place(x=50, y=700)

    # defining yolo real time camera detection buttom
    yoloDetectImageButton = tk.Button(app, text="Detect Image (Yolo)", command=yoloRealTimeDetection)
    yoloDetectImageButton.place(x=250, y=700)

    # defining ssd detect Image buttom
    ssdDetectImageButton = tk.Button(app, text="Detect Image (SSD)", command=ssdPhotoDetection)
    ssdDetectImageButton.place(x=450, y=700)


    # defining ssd real time camera detection buttom
    ssdDetectImageButton = tk.Button(app, text="Detect Image (SSD)", command=ssdRealTimeDetection)
    ssdDetectImageButton.place(x=650, y=700)

    app.mainloop()

