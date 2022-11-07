import torch
import torch.nn as nn
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FontData
from torch.nn import functional as F
from torchvision import transforms
from SwordNet import SwordNet
from tqdm import tqdm
import json

random_seed = 42


def train_and_valid(model,
                    loss_function,
                    dataset,
                    lr=0.0001,
                    val_percent=0.2,
                    epochs=25,
                    bs=128):
    # 1.数据集划分成训练集和验证集
    train_size = int((1 - val_percent) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=1)

    # 2.定义训练设备
    # device = torch.device("mps" if torch.cuda.is_available() else "cpu")  # 设备自行判断
    device = torch.device("mps")
    model.to(device)
    loss_function.to(device)

    # 3.定义优化器
    learning_rate = lr
    optimizer = optim.Adam(FontModel.parameters(), lr=learning_rate)

    # 4.定义学习率调整策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize val acc

    # 记录训练历史数据
    history = {'train_loss': [], 'train_acc': [],
               'valid_loss': [], 'valid_acc': [],
               'best_epoch': 0
               }
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()  # 每轮开始时间记录

        model.train()  # 启用 Batch Normalization 和 Dropout

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        with tqdm(total=train_size, desc="Epoch({}/{})_Train".format(epoch + 1, epochs), leave=True) as tbar:
            for i, item in enumerate(train_loader):  # 训练数据
                inputs = item['image'].to(device)
                labels = item['gt'].to(device)

                labels = F.one_hot(labels, num_classes=18).float()
                # 因为这里梯度是累加的，所以每次记得清零
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                loss.backward()

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                labels = [one_label.tolist().index(1) for one_label in labels]  # 找到下标是1的位置
                labels = torch.tensor(labels).to(device)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                tbar.update(bs)
                tbar.set_postfix(loss=loss.item(), acc=acc.item())
                train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():  # 用于通知dropout层和batchnorm层在train和val模式间切换。
            model.eval()  # model.eval()中的数据不会进行反向传播，但是仍然需要计算梯度；
            with tqdm(total=val_size, desc="Epoch({}/{})_Val  ".format(epoch + 1, epochs), leave=True) as tbar:
                for j, item in enumerate(val_loader):  # 验证数据
                    inputs = item['image'].to(device)
                    labels = item['gt'].to(device)
                    labels = F.one_hot(labels, num_classes=18).float()
                    outputs = model(inputs)  # 模型的输出

                    loss = loss_function(outputs, labels)  # 损失计算

                    valid_loss += loss.item() * inputs.size(0)

                    ret, predictions = torch.max(outputs.data, 1)  # 在分类问题中,通常需要使用max()函数对tensor进行操作,求出预测值索引。
                    # dim是max函数索引的维度0 / 1，0是每列的最大值，1是每行的最大值
                    # 在多分类任务中我们并不需要知道各类别的预测概率，所以第一个tensor对分类任务没有帮助，而第二个tensor包含了最大概率的索引，所以在实际使用中我们仅获取第二个tensor即可。
                    labels = [one_label.tolist().index(1) for one_label in labels]  # 找到下标是1的位置
                    labels = torch.tensor(labels).to(device)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    tbar.update(bs)
                    tbar.set_postfix(loss=loss.item(), acc=acc.item())
                    valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_size
        avg_train_acc = train_acc / train_size

        avg_valid_loss = valid_loss / val_size
        avg_valid_acc = valid_acc / val_size

        # 学习率调整策略
        avg_valid_acc = torch.tensor(avg_valid_acc)
        scheduler.step(avg_valid_acc)
        avg_valid_acc = avg_valid_acc.item()

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['valid_loss'].append(avg_valid_loss)
        history['valid_acc'].append(avg_valid_acc)
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            history['best_epoch'] = best_epoch

        epoch_end = time.time()

        print(
            "\nEpoch: {:03d}\n"
            "Training:   Loss: {:.4f};\n"
            "            Accuracy: {:.4f}%;\n"
            "Validation: Loss: {:.4f};\n"
            "            Accuracy: {:.4f}%;\n"
            "Time Spent: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        if best_epoch == epoch + 1:
            torch.save(model.state_dict(), './checkpoint/' + 'model_best.pt')

    return model, history


if __name__ == '__main__':
    startTime = time.time()
    # 1.定义数据集
    # 定义数据增强策略
    transform = transforms.Compose([
        transforms.ToTensor()  # 将图片转换为Tensor
    ])
    FontDataset = FontData(image_root='./FontData/train', m_transform=transform,
                           mode='train')

    # 2.定义模型结构
    FontModel = SwordNet(3, 18)

    # 3.定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4.定义学习率,batch_size,epoch等超参数
    learningRate = 0.00006
    Epoch = 30
    Batch_size = 128

    # 5.开始训练
    _, train_log = train_and_valid(FontModel, criterion, FontDataset, lr=learningRate, epochs=Epoch)

    # 6.保存训练日志
    with open('./log/'+str(datetime.fromtimestamp(int(startTime)))+".json", "w") as f:
        f.write(json.dumps(train_log, ensure_ascii=False, indent=4, separators=(',', ':')))
    print('finish')
