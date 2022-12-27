### ACPL

Note that our code is built based on [Frozen-in-time](https://github.com/m-bain/frozen-in-time).

### Dependencies

* **Ubuntu 16.04**
* **CUDA 11.3.1**
* **Python 3.8**
* **Pytorch 1.12.0**
* **Timm 0.5.4**
* **transformers 4.17.0**
* **Numpy 1.21.2**
* **Scipy 1.8.0**

### How to use

1. Create data / experiment folders `mkdir data; mkdir exps`. 

   

2. Download the MSR-VTT dataset with the following script, `wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data`.

   

3. Download the pretrained model from Frozen-in-time [here](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid-2m-coco_stformer_b_16_224.pth.tar). Then put the model in the `pretrained/` folder.

   

4. Download the commonsense graph and pretrained swin transformer model ([here](https://pan.baidu.com/s/1coRgWjA2zts4kkXYYXn0LQ) with extract code 5yg5). Then put them in the root of our code.

   

4. Change the configs like `num_gpus` in the config file accordingly.

   

5. Train `python train.py --config configs/msrvtt_jsfusion.json`. Other config files are for different split of dataset.

   

5. Test `python test.py --resume exps/models/{EXP_NAME}/{EXP_TIMESTAMP}/model_best.pth`