# DL-intro-1

## 任务一：**在** **Tiny-ImageNet** **数据集上训练** **Resnet** 模型

### 调试过程

- 尝试使用dummy功能初看脚本是否能运行，结果说是在cpu上运行，太过缓慢。
- 加入指令`--gpu 0`，仍然在cpu上运行。经过查询，`torch.cuda.is_available()`显示false，原因是当初下载是用的清华镜像，应该在官网下载对应cuda版本。
- 使用命令`python HW2.py -a resnet18 -b 64 --epochs 15 --gpu 0 C:\Users\Adydio\Desktop\大二下\pythondl\tiny-imagenet-200\tiny-imagenet-200`没有使用tensorboard，训练速度较慢，acc后面到了0.6,0.7左右。
- 配置了tensorboard，先试一下3个epoch，可以在tensorboard里面看到网络的结构（见后）。手动添加代码记录accuracy和loss，图像均有显示。
- 
