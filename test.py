from model import resnet50
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf

im_height = 224
im_width = 224

# load pic
img = Image.open("H://Project/21ACB/ACB_pretrained/cat_data_test/test/Abyssinian_test_2.jpg")
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
# plt.xticks([])
# plt.yticks([])
# plt.title('The predicted breed of the cat is '+class_indict[str(predict_class)])
# plt.show()