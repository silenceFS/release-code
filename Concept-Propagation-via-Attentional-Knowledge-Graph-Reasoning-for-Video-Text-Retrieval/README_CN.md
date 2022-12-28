## Concept Propagation via Attentional Knowledge Graph Reasoning for Video-Text Retrieval

我们的代码建立在[Frozen-in-time](https://github.com/m-bain/frozen-in-time)的基础上。

### 环境依赖

* **Ubuntu 16.04**
* **CUDA 11.3.1**
* **Python 3.8**
* **Pytorch 1.12.0**
* **Timm 0.5.4**
* **transformers 4.17.0**
* **Numpy 1.21.2**
* **Scipy 1.8.0**

### 数据集

我们的实验建立在公共数据集MSR-VTT上，因为我们采用的基线模型是[Frozen-in-time](https://github.com/m-bain/frozen-in-time)，所以我们的数据集也与它相同，[下载链接](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip)。

### 代码运行步骤

1. 创建存放数据和实验结果的文件夹 `mkdir data; mkdir exps`

   

2. 下载MSR-VTT数据集 `wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data`

   

3. 从[这里](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid-2m-coco_stformer_b_16_224.pth.tar)下载预训练模型，然后将模型放在 `pretrained/` 目录下

   

3. 下载我们构建的常识图谱以及预训练的swin-transformer模型参数 ([百度网盘](https://pan.baidu.com/s/1coRgWjA2zts4kkXYYXn0LQ) ，提取码为 5yg5)，并将它们放在代码的根目录下

   

4. 在config文件中改变如 `num_gpus` 之类的参数。config文件在 `configs/` 目录下，默认为我们最佳性能的实验参数

   

5. 训练 `python train.py --config configs/msrvtt_jsfusion.json`。不同的config文件代表数据集的不同划分

   

5. 测试 `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`
