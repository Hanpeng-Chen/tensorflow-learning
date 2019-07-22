# 调用face_recognition.face_landmarks()方法得到人脸特征点
import face_recognition
import cv2
from matplotlib import pyplot as plt

imagePath = './test_face_landmarks.jpg'

image = face_recognition.load_image_file(imagePath)
face_landmarks_list = face_recognition.face_landmarks(image)

for each in face_landmarks_list:
  for i in each.keys():
    for any in each[i]:
      image = cv2.circle(image, any, 2, (0, 255, 0), 1)

plt.title("Face landmarks")
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()