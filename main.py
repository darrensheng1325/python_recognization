import cv2

image_path = "face_image.jpg"
window_name = f"Detected Objects in {image_path}"
original_image = cv2.imread(image_path)

# Convert the image to grayscale for easier computation
image_grey = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

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

# Draw rectangles on eyes
if len(detected_eyes) != 0:
    for (x, y, width, height) in detected_eyes:
        cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 2)
# Draw rectangles on eyes
if len(detected_face) != 0:
    for (x, y, width, height) in detected_face:
        cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (255, 0, 0), 2)
        
# Draw rectangles on eyes
if len(detected_smile) != 0:
    for (x, y, width, height) in detected_smile:
        cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (0, 0, 255), 2)
if len(detected_cat) != 0:
    for (x, y, width, height) in detected_cat:
        cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (255, 0, 255), 2)
if len(detected_profileface) != 0:
    for (x, y, width, height) in detected_profileface:
        cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (0, 255, 255), 2)

cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
cv2.imshow(window_name, original_image)
cv2.resizeWindow(window_name, 400, 400)
cv2.waitKey(0)
cv2.destroyAllWindows()