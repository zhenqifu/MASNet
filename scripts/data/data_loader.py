import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def transform(train_size):
    return transforms.Compose([
        transforms.Resize((train_size, train_size)),
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])


def transform_sod():
    return transforms.Compose([
        transforms.ColorJitter(0.5,0.5,0.5,0.5)
        # transforms.RandomApply([transforms.ColorJitter(0.5,0.5,0.5,0.5)], p=0.8)
    ])


def transform_test(test_size):
    return transforms.Compose([
        transforms.Resize((test_size, test_size)),
        transforms.ToTensor()
    ])


class Dataset_train(Dataset):
    def __init__(self, img_name_list, lbl_name_list, train_size):
        super(Dataset_train, self).__init__()
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.train_size = train_size
        self.transform = transform(self.train_size)
        self.sod = transform_sod()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert('RGB')
        label = Image.open(self.label_name_list[idx]).convert("L")

        label = label.resize((image.size[0], image.size[1]))


        im_white = Image.new('RGB', (image.size[0], image.size[1]), (0, 0, 0))
        object = Image.composite(image, im_white, label) 

        object_sod = self.sod(object)

        image_sod = Image.composite(object_sod, image, label) 

        # img_name = self.image_name_list[idx].split("/")[-1]
        # image_sod.save('../img_sod/'+ img_name)

        if self.transform:
            seed = np.random.randint(2022)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.transform(label)
            random.seed(seed)
            torch.manual_seed(seed)
            image_sod = self.transform(image_sod)

            # image.save('../img_o/'+ img_name) 
            # label.save('../lab_o/'+ img_name)      


        return image, label, image_sod

class Dataset_test(Dataset):
    def __init__(self, img_name_list, lbl_name_list, test_size):
        super(Dataset_test, self).__init__()
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.test_size=test_size
        self.transform = transform_test(self.test_size)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_name_list[idx]).convert('RGB')
        label = Image.open(self.label_name_list[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


def train_loader(tra_image_list, tra_lbl_list, batch_size_train, train_img_size):
    self_dataset = Dataset_train(img_name_list=tra_image_list, lbl_name_list=tra_lbl_list, train_size=train_img_size)
    train_dataloader = DataLoader(self_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8, drop_last=True)
    return train_dataloader


def test_loader(test_image_list, batch_size_val, test_img_size):
    self_dataset_test = Dataset_test(img_name_list=test_image_list, lbl_name_list=test_image_list, test_size=test_img_size)
    test_dataloader = DataLoader(self_dataset_test, batch_size=batch_size_val, shuffle=False, num_workers=8, drop_last=False)
    return test_dataloader


