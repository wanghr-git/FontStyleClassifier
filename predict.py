# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from SwordNet import SwordNet

# 支持中文
plt.rcParams['font.sans-serif'] = ['STHeiti']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

random_seed = 42
device = torch.device('cpu')


def predict(model, img_path, save_path):
    image = Image.open(img_path)
    tran = transforms.ToTensor()
    inputs = tran(image)
    inputs = torch.unsqueeze(inputs, dim=0).float()
    model = model.to(device)
    model.eval()
    outputs = model(inputs)
    prop, predicted = torch.max(outputs, 1)
    prop = prop.item()
    predicted = predicted.item()
    labels = ['古文字形', '姚体', '彩云', '柳公权', '楷体', '欧阳询', '汉隶书', '琥珀', '米芾行书', '舒体', '行书',
              '行楷', '行草', '说文小篆', '隶书', '颜真卿勤礼碑', '颜真卿多宝塔碑', '黑体']
    pred_label = labels[predicted]
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('class:' + pred_label + '\tprop:' + str(format(prop, '.4f')))  # 图像题目
    plt.savefig(save_path)  # 保存图片
    plt.show()
    print('jj')


if __name__ == '__main__':
    # 1.定义模型
    net = SwordNet(3, 18)

    # 2.加载训练好的模型参数
    ckpt_path = './checkpoint/model_best.pt'    # 训练好的模型参数地址
    net.load_state_dict(torch.load(ckpt_path), False)

    # 3.预测图片并保存预测结果
    img_to_pred = './FontData/val/彩云/严.png'  # 需要预测的图片
    pred_save_path = './prediction/严.png'         # 预测结果的保存地址
    predict(model=net, img_path=img_to_pred, save_path=pred_save_path)
