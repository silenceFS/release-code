# Learning Linguistic Association Towards Efficient Text-Video Retrieval

LINAS可以应用不同的基线模型，我们这里提供的是基于“Dual Encoding”的示例代码。

## 配置环境

* **Ubuntu 16.04**
* **CUDA 11.1**
* **Python 3.8**
* **Pytorch 1.9.0**
* **Numpy 1.19.2**
* **Scipy 1.5.4**
* **Tensorboard-logger 0.1.0**

## 数据集

在跑我们的代码之前，请先下载代码所需要的数据集以及预训练的word2vec模型（[下载地址](https://drive.google.com/drive/folders/1TEIjErztZNQAi6AyNu9cK5STwo74oI8I)）。然后将它们解压到code/dataset/目录下。
我们用的是公开数据集MSR-VTT。由于我们的基线模型来自于Dong et al [Dual Encoding for Video Retrieval by Text]，所以我们直接用他们公开的处理后的数据。

## 训练

跑以下的命令来完成训练。

我们的最好性能是在support-set-size设置为8时得到的。

```shell
cd code
./train_all.sh $GPU_DEVICE $support-set-size
```

## 评估

在测试集上的性能评估会在训练结束后自动进行。如果你希望在训练结束前进行评估，那么在停止进程后，运行以下命令。

```shell
cd code
./evaluate.sh $GPU_DEVICE $support-set-size
```


## 适应性蒸馏

将 ***train_all.sh*** 中的 '***--similarity_type***' 从 '***diag***' 改为 '***adapt***' 来应用我们的适应性蒸馏策略，学习可动态变化的mask矩阵。

