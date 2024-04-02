import cv2
import numpy as np
from os.path import dirname, join

# Paths for the model files
prototxt_path = join(dirname(__file__), "deploy.prototxt")
model_path = join(dirname(__file__), "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load the Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Replace with your RTSP stream URL
rtsp_url = 'rtsp://172.31.0.182:8554/100e3af7'
cap = cv2.VideoCapture(rtsp_url)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
        # Reduce the resolution of the captured frame
    frame = cv2.resize(frame, (640, 480))  # Adjust the resolution as needed

    # Get width and height of the frame
    h, w = frame.shape[:2]
    # Gaussian blur kernel size depends on width and height of the frame
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    
    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    output = np.squeeze(model.forward())

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        if confidence > 0.2:
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(int)
            # Blur the face in the frame
            face = frame[start_y: end_y, start_x: end_x]
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            frame[start_y: end_y, start_x: end_x] = face

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
