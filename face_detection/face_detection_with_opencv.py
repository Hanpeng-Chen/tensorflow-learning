# 使用OpenCV进行人脸检测
import cv2
from matplotlib import pyplot as plt

imagePath = './test_face_detection.jpg'
# 引入OpenCV提供的人脸分类模型xml
cascPath = './haarcascade_frontalface_default.xml'
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# 读取图像并转为灰度图
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测图片中的人脸
faces = faceCascade.detectMultiScale(
  gray,
  scaleFactor=1.1,
  minNeighbors=4,
  minSize=(30, 30)
)
color = (0, 255, 0)
# 用矩形框将人脸框出来
for (x, y, w, h) in faces:
  cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

plt.title("Found {0} faces!".format(len(faces)))
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()