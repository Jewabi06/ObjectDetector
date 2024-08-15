```python
import cv2

# Load the YOLOv4 Tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class names from file
classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize video capture from the webcam
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Read a frame from the video capture
    ret, frame = video.read()

    # Detect objects in the frame
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        # Draw the class name and bounding box on the frame
        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    # Display the frame with detected objects
    cv2.imshow("Detector", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and destroy all OpenCV windows
video.release()
cv2.destroyAllWindows()
