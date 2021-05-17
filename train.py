from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pit
from model import resnet101
import tensorflow as tf
import json
import os
import PIL.Image as im
import numpy as np

# get source path
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
# get cat image path
# TODO: path
image_root = "H://Project/21ACB/ACB_pretrained/cat_faces/"

# train, validation, test path
train_dir = image_root + "train"
validation_dir = image_root + "validation"

# TODO: num_classes
num_classes = 11
# define train parameters
im_height = 224
im_width = 224
batch_size = 16
epochs = 24

# TODO: preprocessing of images

# load train datasets
train_image_generator = ImageDataGenerator(horizontal_flip=True)
train_data_generator = train_image_generator.flow_from_directory(directory=train_dir,
                                                                 batch_size=batch_size,
                                                                 shuffle=True,
                                                                 target_size=(im_height, im_width),
                                                                 class_mode='categorical')
# count total number of train sample
total_train = train_data_generator.n
# load validation datasets
validation_image_generator = ImageDataGenerator(horizontal_flip=True)
validation_data_generator = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                           batch_size=batch_size,
                                                                           shuffle=True,
                                                                           target_size=(im_height, im_width),
                                                                           class_mode='categorical')
# count total number of validation
total_validation = validation_data_generator.n

# 获得类别字典
class_indices = train_data_generator.class_indices
# 转换类别字典中键和值的位置
inverse_dict = dict((val, key) for key, val in class_indices.items())
# 将数字标签字典写入json文件：class_indices.json
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# TODO: update the num_classes
feature = resnet101(num_classes=num_classes, include_top=False)
# feature.build((None, 224, 224, 3))  # when using subclass model
feature.load_weights('pretrain_weights.ckpt')  # 加载预训练模型
feature.trainable = False  # 训练时冻结与训练模型参数
feature.summary()  # 打印预训练模型参数

# 在原模型后加入两个全连接层，进行自定义分类
# TODO: update the tf.keras.layers.Dense
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(num_classes),
                             tf.keras.layers.Softmax()])
# model.build((None, 224, 224, 3))
model.summary()  # 打印增加层的参数

# 模型装配
# 1.目标损失函数：交叉熵
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# 2.优化器：Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
# 3.评价标准：loss和accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:  # 建立梯度环境
        output = model(images, training=True)  # 前向计算
        loss = loss_object(labels, output)  # 计算目标损失
    gradients = tape.gradient(loss, model.trainable_variables)  # 自动求梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 反向传播，更新参数
    train_loss(loss)  # 计算训练集的平均损失值loss
    train_accuracy(labels, output)  # 计算训练集上的准确性accuracy


@tf.function
def test_step(images, labels):
    output = model(images, training=False)  # 前向计算
    t_loss = loss_object(labels, output)  # 求每一次的目标损失值
    test_loss(t_loss)  # 求平均损失值
    test_accuracy(labels, output)


best_test_loss = float('inf')
for epoch in range(1, epochs + 1):
    # 重置清零每一轮的loss值和accuracy。因此后面打印出来的是每一个epoch中的loss、accuracy平均值，而不是历史平均值。
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    # 训练集训练过程
    for step in range(total_train // batch_size):  # 一个epoch需要迭代的step数
        images, labels = next(train_data_generator)  # 一次输入batch_size组数据
        train_step(images, labels)  # 训练过程

        # 打印训练过程
        rate = (step + 1) / (total_train // batch_size)  # 一个epoch中steps的训练完成度
        a = "*" * int(rate * 50)  # 已完成进度条用*表示
        b = "." * int((1 - rate) * 50)  # 未完成进度条用.表示
        acc = train_accuracy.result().numpy()
        print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
    print()

    # 验证集测试过程
    for step in range(total_validation // batch_size):
        test_images, test_labels = next(validation_data_generator)
        test_step(test_images, test_labels)  # 在验证集上测试，只进行前向计算
    #  每训练完一个epoch后，打印显示信息
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()
        # 保存模型参数
        model.save_weights("./save_weights/resNet_101.ckpt", save_format="tf")
