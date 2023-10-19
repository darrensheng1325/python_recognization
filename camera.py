import numpy as np
import cv2
import time
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
from threading import Thread
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
    image_bgr = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0)
    image_small = cv2.resize(image_bgr, (128, 64))
    # Display the resulting frame
    cv2.imshow('frame', cv2.flip(frame, 0))
    if cv2.waitKey(1) == ord('q')or True:
        cv2.imwrite('capture.ppm', image_small)
        #image = Image.open('capture.ppm').convert('1')
        #disp.image(image)
        #disp.display()
    cv2.imwrite('image.ppm', background)
    image = Image.open('image.ppm').convert('1')
    image_grey = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 0)
def capture():
    global disp
    while True:
        predict()
        
def predict():
    img = "capture.ppm"
    path = "/home/pi/python_recognization/face_image.jpg"
    # img_array = imread(os.path.join(path,img))
    img_array = Image.open(path)
    img_shown=False
    flat_data = []
    images = []
    target = []
    target_class = []
    img_idx=0
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    if img_array.mode != 'RGB':
        img_array = img_array.convert('RGB')
    img_array = np.array(img_array)
    if not img_shown:
      print('showing image ', img_idx)
      # plt.imshow(img_array)
      axs[img_idx].imshow(img_array)
      img_idx += 1
      img_shown = True
    # Skimage normalizes the value of image
    img_resized = resize(img_array,(150,150,3))[0]
    flat_image = img_resized.flatten()
    print(flat_image.shape)
    flat_data.append(flat_image)
    images.append(img_resized)
    target.append(target_class)
    # Convert list to numpy array format
    flat_data = np.array(flat_data)
    images = np.array(images)
    target = np.array(target)
    df = pd.DataFrame(flat_data)
    # Create a column for output data called Target
    #df['Target'] = target
    # Rows are all the input images (90 images, 30 of each category)
    print(df)
    model = load("model.joblib")
    prediction = model.predict(df)
    if prediction == [1]:
        image = Image.open("pose_images/dance_0.ppm")
    else:
        image = Image.open("background.ppm")
    disp.display()
    
Thread(capture).start()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()