from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class FontData(Dataset):
    def __init__(self, image_root, m_transform, mode):
        self.image_root = image_root  # 图像根目录，也就是各类图片存放地址的上一层
        self.transform = m_transform  # 图像增强方式集合
        self.mode = mode              # 当前模型运行模式：train,val,test
        self.Data = ImageFolder(self.image_root, transform=self.transform)
        self.len = len(self.Data)

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            image = self.Data[item][0]
            gt = self.Data[item][1]
            sample = {'image': image, 'gt': gt}
            return sample
        else:
            image = self.Data[item][0]
            sample = {'image': image}
            return sample

    def __len__(self):
        return self.len
