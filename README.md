# Cognitive
认知科学作业

## 进度条
- baseline
- EGGNet
- TSception
- visdom可视化

## 使用说明
- models : 存放写好了的网络模型或者其他模型
- utils:工具和配置
	- options:全局配置
	- utils:工具， 包括:加载数据，训练，保存和加载权重，etc
- methods:方法， baseline, etc
- datasets:存放数据，处理数据
-

## Examples
1. 使用 python -m visdom.server 打开端口

2. 运行
```python
python main.py --model TSception
```

3. 打开浏览器 localhost:8888， 查看训练过程
