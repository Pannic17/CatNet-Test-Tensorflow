# CatNet-Test-Tensorflow

CatNet 早期 TensorFlow 猫品种分类与数据处理实验。当前 `model.py` 实现的是 **ResNet-34 / ResNet-50 / ResNet-101**；训练入口使用 ResNet-101，并非旧 README 所述的 MobileNetV2。

主项目入口：[CatNet-Unity](https://github.com/Pannic17/CatNet-Unity)。后续 7 类毛色／花纹的 MobileNet 实验见 [CatNet-Tensorflow](https://github.com/Pannic17/CatNet-Tensorflow)。

## 内容

| 文件 | 用途 |
| --- | --- |
| `model.py` | 残差模块和 ResNet 网络定义 |
| `train.py` | 冻结 ResNet-101 特征网络，训练附加分类层 |
| `test.py` | 单图预测及猫脸裁剪辅助函数；当前构建 ResNet-50 |
| `reweight.py` | 将外部 ResNet-101 checkpoint 变量重命名，生成 `pretrain_weights.ckpt` |
| `class_indices.json` | 11 个品种标签 |
| `prev.py` | TensorFlow Datasets `cats_vs_dogs` 数据集加载实验 |
| `rename_image.py` | 交互式批量重命名为 JPG 文件名 |
| `unpack_tar_gz.py` | 解压本地 `annotations.tar.gz` |
| `shutterstock_spider.py` | 历史网页缩略图下载实验，与分类训练入口独立 |

类别为 Abyssinian、Bengal、Birman、Bombay、British_Shorthair、Egyptian_Mau、Maine_Coon、Persian、Ragdoll、Russian_Blue、Siamese。

## 环境与数据

主训练／预测脚本使用 Python、TensorFlow、NumPy、Pillow、Matplotlib；`test.py` 还使用 OpenCV。可在独立环境中安装这些依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install tensorflow numpy pillow matplotlib opencv-python
```

`prev.py` 额外使用 `tensorflow-datasets`，下载实验使用 `requests`。仓库没有锁定依赖版本；代码含历史 TensorFlow/Keras API 和旧 TFDS split API，安装后需检查兼容性，以上命令不代表已验证的版本组合。

准备按品种分类的数据目录，并将 `train.py` 的 `image_root` 改为本地路径：

```text
<image_root>/
  train/<品种名>/*.jpg
  validation/<品种名>/*.jpg
```

训练数据集、`resnet_v1_101.ckpt`、`pretrain_weights.ckpt` 和训练输出权重均未随仓库提供。

## 训练与预测

1. 准备与 `reweight.py` 变量命名匹配的 ResNet-101 预训练 checkpoint，修改路径后执行 `python reweight.py`。
2. 检查数据路径和类别数量，创建 `save_weights` 目录，从仓库根目录执行 `python train.py`。默认输入为 224 × 224，batch size 为 16，训练 24 轮，Adam 学习率为 0.0002。
3. 训练会重写 `class_indices.json`，按更低的验证损失保存 `save_weights/resNet_101.ckpt`。
4. 预测前修改 `test.py` 的图片和级联 XML 路径，并先统一训练与预测的网络结构，再执行 `python test.py`。

**当前训练／预测结构不一致：** `train.py` 使用 ResNet-101，`test.py` 使用 ResNet-50 却加载 `resNet_101.ckpt`。不能将现有预测脚本视为与训练输出直接兼容。该分支也没有采用主项目的 `[-1, 1]` 像素归一化流程。

批量重命名会直接修改文件名；运行这些数据工具前请检查目标目录。本文基于源码说明，未执行训练、数据下载或模型推理。
