import os
import tensorflow as tf
import numpy as np
import cv2

face_mask = ['Masked', 'No mask']
size = 224

# Load face detection and face mask model
model = tf.keras.models.load_model("face_mask.model")
faceNet = cv2.dnn.readNet('computer_vision-master/CAFFE_DNN/deploy.prototxt.txt',
                          'computer_vision-master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (size, size))
        face = np.reshape(face, (1, size, size, 3)) / 255.0
        result = np.argmax(model.predict(face))

        if result == 0:
            label = face_mask[result]
            color = (0, 255, 0)
        else:
            label = face_mask[result]
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
        cv2.rectangle(frame, (startX, startY - 60), (endX, startY), color, -1)
        cv2.putText(frame, label, (startX + 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 800, 600)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()