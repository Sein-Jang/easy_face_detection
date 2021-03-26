# Author: Sein Jang

import cv2
import time

from retinaface import RetinaFace

detector = RetinaFace("path to retinaface weights", False, 0.4)

# capture frame from video
cap = cv2.VideoCapture('sample_video')

cap.set(3, 480)
cap.set(4, 480)

width = int(cap.get(3))
height = int(cap.get(4))
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('file name of output result', fcc, 30, (width, height))

prevTime = 0

fps_list = []

while True:
    ret, frame = cap.read()

    # convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # detect face
    faces, face_locations = detector.detect(frame, 0.9)
    
    # Draw rectangle around the faces
    for x, y, w, h, confidence in faces:
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)

    curTime = time.time()

    sec = curTime - prevTime

    prevTime = curTime

    fps = 1/(sec)
    fps_list.append(fps)

    fps_on_frame = "FPS : %0.1f" % fps

    cv2.putText(frame, fps_on_frame, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

    # write video
    out.write(frame)

    # display the result
    # cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()