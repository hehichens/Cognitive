![](./pic/logo3.png)

## 目录

[toc]

## 进度条
- baseline:white_check_mark:
- EGGNet:white_check_mark:
- TSception:white_check_mark:
- visdom可视化:white_check_mark:

## 使用说明
- models : 存放写好了的网络模型或者其他模型
- utils:工具和配置
	- options:全局配置
	- utils:工具， 包括:加载数据，训练，保存和加载权重，etc
- methods:方法， baseline, etc
- datasets:
  - data:存放数据
  - *.oy:处理数据



## 数据

[DEAP ](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html)

被试：32名（16名男性和16名女性）；

EEG电极：10-20国际标准的32导联电极帽；

采样率：512Hz;

在观看完1分钟的刺激视频后，被试按照从1～9的大小，标记所看视频的效价(Valence)、唤醒度(Arousal)、优势度(Dominance)和喜爱度(Liking)的大小。

常利用预处理(降采样128Hz，4-45Hz带通滤波，去除眼电)后的脑电数据。

![](./pic/data.png)

## 网络结构

### EEGNet

from the paper: EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

![](./pic/EEGNet.png)

### TSception

from the paper: TSception:A Deep Learning Framework for Emotion Detection Using EEG

![](./pic/TSception.png)



## Examples

### 数据处理

```python
# for small data (40, 101)
python split.py --data_path ./data/small_data.npy

# for big data (40, 7680)
python split.py --data_path ./data/data.npy
```

得到

```
train_data.npy
train_label.npy
test_data.npy
test_label.npy
```





### 训练

1. 打开 visdom 
```
python -m visdom.server  port 8888
```

打开端口 8888

2. 运行
```python
# see the utils/options,py for more parameters 
python main.py --model basemodel --num_epochs 400
```

3. 打开浏览器 localhost:8888， 查看训练过程



## 结果

### Small data

**特征处理**

将每个通道的8064个数据划分为10段，计算每段的平均值、中值、最大值、最小值、标准差、方差、范围、偏度和峰度值这9个统计特征，然后加上整体8064个数据的9个统计特征，实验编号和参与者编号。 8064 -> 9*10+11=101 （40，40， 8064）-> （40，40，101）

| Model|Epochs | Learning rate | Accuracy |
| - | - | - | - |
| | | | |




## Contact me 
:heart::heart::heart::wink:
- hichens@qq.com
- hichens1559576916@gmail.com