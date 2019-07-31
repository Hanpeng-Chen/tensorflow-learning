# 加载训练数据集
import numpy as np
import cv2
import os.path

class IdentityMetadata():
    def __init__(self, base, file):
        self.base = base # 数据集根目录
        # self.name = name # 目录名
        self.file = file # 图像文件名

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.file) 

def load_metadata(path):
    metadata = []
    for f in os.listdir(path):
        # 检查文件名后缀，仅支持 jpg 和 jpeg 两种文件格式
        ext = os.path.splitext(f)[1]
        if ext == '.jpg' or ext == '.jpeg':
            metadata.append(IdentityMetadata(path, f))
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV 默认使用 BGR 通道加载图像，转换为 RGB 图像
    return img[...,::-1]

metadata = load_metadata('images')



# 人脸检测、对齐和提取
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib

# 初始化 OpenFace 人脸对齐工具，使用 Dlib 提供的 68 个关键点
alignment = AlignDlib('landmarks.dat')


# 加载一张训练图像
img = load_image(metadata[0].image_path())
# 检测人脸并返回边框
bb = alignment.getLargestFaceBoundingBox(img)
# 使用指定的人脸关键点转换图像并截取 96x96 的人脸图像
aligned_img = alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
# 绘制原图
# plt.figure(1)
# plt.subplot(131)
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# # 绘制带人脸边框的原图
# plt.subplot(132)
# plt.imshow(img)
# plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
# plt.xticks([])
# plt.yticks([])
# # 绘制对齐后截取的 96x96 人脸图像
# plt.subplot(133)
# plt.imshow(aligned_img)
# plt.xticks([])
# plt.yticks([])
# plt.show()



# # 加载nn4.small2.v1模型
# from model import create_model
# nn4_small2 = create_model()
# from keras.models import Model
# from keras.layers import Input, Layer

# # 输入 anchor, positive and negative 96x96 RGB图像
# in_a = Input(shape=(96, 96, 3))
# in_p = Input(shape=(96, 96, 3))
# in_n = Input(shape=(96, 96, 3))

# # 输出对应的人脸特征向量
# emb_a = nn4_small2(in_a)
# emb_p = nn4_small2(in_p)
# emb_n = nn4_small2(in_n)

from model import create_model

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('models/nn4.small2.v1.h5')


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # 数据规范化
    img = (img / 255.).astype(np.float32)
    # 人脸特征向量
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


# Squared L2 Distance
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

count = 0
def show_pair(idx1, idx2):
    global count
    count += 1
    plt.figure(num=count, figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    plt.xticks([])
    plt.yticks([])


show_pair(0, 1)
show_pair(0, 2)
show_pair(1, 2)
plt.show()