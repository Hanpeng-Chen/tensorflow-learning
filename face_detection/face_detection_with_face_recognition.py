# 使用face_recognition库进行人脸检测
import face_recognition
import cv2
from matplotlib import pyplot as plt

imagePath = './test_face_detection.jpg'

# 使用face_recognition加载图片,并检测人脸
image = face_recognition.load_image_file(imagePath)
#检测图片中所有人脸
face_locations = face_recognition.face_locations(image)

# 用矩形框框出检测到的人脸
for (top, right, bottom, left) in face_locations:
  cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

plt.title("Found {0} faces!".format(len(face_locations)))
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()