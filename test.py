import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from SwordNet import SwordNet
from dataset import FontData

random_seed = 42


def test(model,
         dataset,
         bs=64):
    test_loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=1)
    test_size = len(dataset)

    # 2.定义训练设备
    # device = torch.device("mps" if torch.cuda.is_available() else "cpu")  # 设备自行判断
    device = torch.device("mps")
    model.to(device)
    test_acc = 0.0
    model.eval()  # model.eval()中的数据不会进行反向传播，但是仍然需要计算梯度；
    with tqdm(total=test_size, desc="test", leave=True) as tbar:
        for j, item in enumerate(test_loader):  # 验证数据
            inputs = item['image'].to(device)
            labels = item['gt'].to(device)
            labels = F.one_hot(labels, num_classes=18).float()
            outputs = model(inputs)  # 模型的输出

            ret, predictions = torch.max(outputs.data, 1)  # 在分类问题中,通常需要使用max()函数对tensor进行操作,求出预测值索引。
            # dim是max函数索引的维度0 / 1，0是每列的最大值，1是每行的最大值
            # 在多分类任务中我们并不需要知道各类别的预测概率，所以第一个tensor对分类任务没有帮助，而第二个tensor包含了最大概率的索引，所以在实际使用中我们仅获取第二个tensor即可。
            labels = [one_label.tolist().index(1) for one_label in labels]  # 找到下标是1的位置
            labels = torch.tensor(labels).to(device)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            tbar.update(bs)
            tbar.set_postfix(acc=acc.item())
            test_acc += acc.item() * inputs.size(0)

    avg_test_acc = test_acc / test_size

    return avg_test_acc


if __name__ == '__main__':
    startTime = time.time()
    # 1.定义数据集
    # 1.1定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor()  # 将图片转换为Tensor
    ])
    # 1.2数据集
    FontDataset = FontData(image_root='./FontData/val', m_transform=transform, mode='val')

    # 2.定义模型结构
    FontModel = SwordNet(3, 18)

    # 3.加载训练好的模型
    FontModel.load_state_dict(torch.load('./checkpoint/model_best.pt'), False)

    # 4.开始评价精度
    Top1_Acc = test(model=FontModel, dataset=FontDataset)

    print('Top1_Acc:', Top1_Acc)
