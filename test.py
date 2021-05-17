from model import resnet50
from PIL import Image
import cv2
import math
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf

def detect(filename):
    # detect the cat face by opencv haarcascade
    # return the coordinates of cat face
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier(
        'H://Study/Year G/unity_python/std/cat_face_cut/haarcascade_frontalcatface.xml')
    # 读取图片
    img = cv2.imread(filename)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.02,
                                          minNeighbors=5)
    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        w_delta = math.ceil(w * 0.1)
        h_delta = math.ceil(h * 0.1)
        print(w_delta, h_delta)
        print(x, y)

        # change the coordinate, increase the rectangle by 10% for each side and move up to include the ears
        x1 = x - w_delta
        y1 = y - h_delta * 2
        x2 = x + w + w_delta
        y2 = y + h

        # draw the cat face, originate and cut
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        print(x1, y1, x2, y2)
        # return x1, y1, x2, y2
        coordinates = [x1, y1, x2, y2]
        positive = True
        for coordinate in coordinates:
            if coordinate < 0:
                positive = False
                break
        if positive is True:
            return coordinates
    # 命名显示窗口
    # cv2.namedWindow('cat')
    # 显示图片
    # cv2.imshow('cat', img)
    # 保存图片
    # cv2.imwrite('cxks.png', img)
    # 设置显示时间,0表示一直显示
    # cv2.waitKey(0)
    # return x1, y1, x2, y2


def cut(filename):
    # cut the cat face out according to the coordinates given by detection
    # return the image for saving
    coordinates = detect(filename)

    img = cv2.imread(filename)
    if coordinates is not None:
        new = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        return new
    else:
        return None

im_height = 224
im_width = 224

# load pic
img = Image.open("H://Project/21ACB/ACB_pretrained/cat_faces/test/Bombay_test_1.jpg")

# cut the cat face out
# img = cut("H://Project/21ACB/ACB_pretrained/cat_data_test/test/Abyssinian_test_4.jpg")
# resize 224x224
plt.imshow(img)
img = img.resize((im_width, im_height))



# preprocessing /abort

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

# read labels from class_indices.json
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


# adjust resnet model
# TODO: update num_classes
feature = resnet50(num_classes=11, include_top=False)
feature.trainable = False
# TODO: update layers.dense
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(11),
                             tf.keras.layers.Softmax()])

# model.build((None, 224, 224, 3))  # when using subclass model
# load trained model
model.load_weights('./save_weights/resNet_101.ckpt')
result = model.predict(img)
prediction = np.squeeze(result)
predict_class = np.argmax(result)
print('预测该图片类别是：', class_indict[str(predict_class)], ' 预测概率是：', prediction[predict_class])

# plt management
plt.xticks([])
plt.yticks([])
plt.title('The predicted breed of the cat is '+class_indict[str(predict_class)])
plt.show()