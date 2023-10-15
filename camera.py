import numpy as np
import cv2
import time
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
from PIL import Image
cap = cv2.VideoCapture(0)
RST = 24
disp = Adafruit_SSD1306.SSD1306_128_64(rst=RST)
disp.begin()
disp.clear()
disp.display()
background_original = cv2.imread('background.ppm')
background = background_original
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 0)
    # Display the resulting frame
    cv2.imshow('frame', cv2.flip(frame, 0))
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite('image.ppm', background)
        image = Image.open('image.ppm').convert('1')
        disp.image(image)
        disp.display()
        break
    cv2.imwrite('image.ppm', background)
    image = Image.open('image.ppm').convert('1')
    disp.image(image)
    disp.display()
    image_grey = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 0)
    image_bgr = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0)
    image_small = cv2.resize(image_bgr, (128, 64))

    eye_classifier = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_eye.xml")

    face_classifier = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_frontalface_alt.xml")

    smile_classifier = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_smile.xml")

    catface_classifier = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalcatface.xml")

    profileface_classifier = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_profileface.xml")

    detected_eyes = eye_classifier.detectMultiScale(image_grey, minSize=(10, 10))
    detected_face = face_classifier.detectMultiScale(image_grey, minSize=(15, 15))
    detected_smile = smile_classifier.detectMultiScale(image_grey, minSize=(200, 150))
    detected_cat = catface_classifier.detectMultiScale(image_grey, minSize=(5, 5))
    detected_profileface = profileface_classifier.detectMultiScale(image_grey, minSize=(15, 15))
    if len(detected_eyes) != 0:
        for (x, y, width, height) in detected_eyes:
            cv2.rectangle(frame, (x, y),
                          (x + height, y + width),
                          (0, 255, 0), 2)
    # Draw rectangles on eyes
    if len(detected_face) != 0:
        for (x, y, width, height) in detected_face:
            cv2.rectangle(frame, (x, y),
                          (x + height, y + width),
                          (255, 0, 0), 2)
            
    # Draw rectangles on eyes
    if len(detected_smile) != 0:
        for (x, y, width, height) in detected_smile:
            cv2.rectangle(frame, (x, y),
                          (x + height, y + width),
                          (0, 0, 255), 2)
    if len(detected_cat) != 0:
        for (x, y, width, height) in detected_cat:
            cv2.rectangle(frame, (x, y),
                          (x + height, y + width),
                          (255, 0, 255), 2)
    if len(detected_profileface) != 0:
        for (x, y, width, height) in detected_profileface:
            cv2.rectangle(frame, (x, y),
                          (x + height, y + width),
                          (0, 255, 255), 2)

cap = cv2.VideoCapture(0)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()