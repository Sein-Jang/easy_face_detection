# Easy Face Detection
Introduction of simple-to-use face detection models


### 1. openCV - haar cascades
```python
import cv2

# Load model & img
detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
img = cv2.imread('imgs/faces.jpeg')

# Face detection
faces = detector.detectMultiScale(gray, 1.1, 4)

# Grayscale to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

"""
Sample image "faces.jpeg"
- Wall time: 855 ms
- Detected faces: 29
"""
```

### 2. DLib - Histogram of Oriented Gradients (HOG)
```python
import cv2
import dlib

# Load model & img
detector = dlib.get_frontal_face_detector()
img = dlib.load_rgb_image(image_file)

# Face detection
faces = detector(img, 1)

"""
Sample image "faces.jpeg"
- Wall time: 492 ms
- Detected faces: 33
"""
```

### 3. DLib - Convolutional Neural Network (CNN)
```python
import cv2
import dlib

# Load model & img
detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
img = dlib.load_rgb_image(image_file)

# Face detection
faces = detector(img, 1)

"""
Sample image "faces.jpeg" 
- Wall time: 1min 13s
- Detected faces: 41
"""
```
#### Using the above 3 models, you can perform face detection very simply.
![result](imgs/sample_detection.png)


### 4. BlazeFace
```python
import cv2
import dlib

# Load model
net = BlazeFace()
net.load_weights("./blazeface.pth")
net.load_anchors("./anchors.npy")

# Optionally change the thresholds:
net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3

# Load image and resize image to (128, 128)
img = cv2.imread(image_file)
img = cv2.resize(img, (128, 128))

# Face detection
faces = net.predict_on_image(frame)

"""
Sample image "faces.jpeg" 
- Wall time: 108 ms
- Detected faces: 0
- !! BlazeFace doesn't work very well on small faces
"""
```


### 5. RetinaFace

### 6. Multi-task Cascaded Convolutional Networks (MTCNN)
