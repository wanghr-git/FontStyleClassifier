import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from SwordNet import SwordNet

plt.rcParams['font.sans-serif'] = ['STSong']


def get_image_info(m_image_dir):
    m_image_info = Image.open(m_image_dir).convert('RGB')  # 是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    m_image_info = image_transform(m_image_info)  # torch.Size([3, 224, 224])
    m_image_info = m_image_info.unsqueeze(0)
    return m_image_info  # 变成tensor数据


def get_k_layer_feature_map(m_model, k, x):
    with torch.no_grad():
        for index, layer in enumerate(m_model.children()):  # model的第一个Sequential()是有多层，所以遍历
            x = layer(x)  # torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(m_feature_map, layer_name, size=(96, 96)):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    m_feature_map = m_feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
    m_feature_map = m_feature_map.view(1, m_feature_map.shape[0], m_feature_map.shape[1], m_feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=size)  # 这里进行调整大小
    m_feature_map = upsample(m_feature_map)
    m_feature_map = m_feature_map.view(m_feature_map.shape[1], m_feature_map.shape[2], m_feature_map.shape[3])

    feature_map_num = m_feature_map.shape[0]  # 返回通道数
    row_num = int(np.ceil(np.sqrt(feature_map_num)))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
        plt.subplot(row_num, row_num, index)
        plt.imshow(m_feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
    plt.savefig(layer_name)
    plt.show()


def save_all_featureMap(net, img_path, save_path):
    image = get_image_info(img_path)
    image_name = img_path.split('/')[-1].split('.')[0]
    for i in range(len(list(net.children()))):
        m_feature_map = get_k_layer_feature_map(model, i, image)
        if os.path.exists(save_path):
            show_feature_map(m_feature_map, layer_name=os.path.join(save_path, image_name+'_Layer_' + str(i) + '_FeatureMap'))
        else:
            os.makedirs(save_path)
            show_feature_map(m_feature_map, layer_name=os.path.join(save_path, image_name+'_Layer_' + str(i) + '_FeatureMap'))


if __name__ == '__main__':
    image_dir = './FontData/val/行楷/万.png'
    # 定义提取第几层的feature map
    k = 1
    image_info = get_image_info(image_dir)

    # 1.定义模型
    model = SwordNet(3, 18)

    # 2.加载训练好的模型参数
    ckpt_path = './checkpoint/model_best.pt'  # 训练好的模型参数地址
    model.load_state_dict(torch.load(ckpt_path), False)

    feature_map = get_k_layer_feature_map(model, k, image_info)
    # show_feature_map(feature_map, layer_name='Layer_'+str(k)+'_FeatureMap')
    save_all_featureMap(net=model, img_path=image_dir, save_path='./visual')
