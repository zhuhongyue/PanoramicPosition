# -*- coding: utf-8 -*
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import pandas as pd
import re
import pdb
def readData(txt_url,isShuffle=0):
    folder_path = os.path.dirname(txt_url)
    df = pd.read_csv(txt_url)
    image_folder = df['图片文件夹']
    x = list(df['x'])
    y = list(df['y'])
    z = list(df['z'])
    loc = [[a,b,c] for (a,b,c) in zip(x,y,z)]
    img_file = [folder_path+"/"+i for i in image_folder]
    return img_file,loc

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
tensor_preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

SIFT_preprocess = transforms.Compose([
    transforms.ToTensor()
])

def default_loader(path):
    dirs = os.listdir(path)
    img_tensor = []
    img = []
    fn = []
    for i in dirs:                             
        if os.path.splitext(i)[1] == ".jpg" and "origin" in i:   
            img_tensor.append(tensor_preprocess(Image.open(path+"/"+i).resize((224,224))))
            img.append(SIFT_preprocess(Image.open(path+"/"+i)))
            fn.append(path+"/"+i)
    return img_tensor,img,fn

class trainset(Dataset):
    def __init__(self, path):
        self.images, self.target = readData(path,0)
        self.loader = default_loader
        self.path = path
    def __getitem__(self, index):
        path = self.images[index]
        img_feat, img ,fn = self.loader(path)
        target = self.target[index]
        return img_feat,img,target,fn
    def __len__(self):
        return len(self.images)
        
if __name__ == "__main__":
    img_set, loc = readData('/home/zjfan/SmartCityCompaign/dataset/train_data/scene1_jiading_lib_training/scene1_jiading_lib_training_coordinates.csv',0)
    troch_trainset = trainset('/home/zjfan/SmartCityCompaign/dataset/train_data/scene1_jiading_lib_training/scene1_jiading_lib_training_coordinates.csv')
    #load image from certain path
    #for i in img_set:
    #    print i
    trainloader = DataLoader(troch_trainset, batch_size=1,shuffle=True)#do not test other batchsize
    #example for loading data
    for image_feat, img, location, fn in trainloader:
        print "image len", len(image_feat[0]) #一个坐标对应6张图像
        print "image shape:", image_feat[0][0].shape#每张图的张量维度
        print "location shape:", location[0].shape #坐标的维度
        print "image 0 file:", fn[0]  #所在文件夹  
        print "origin img shape",img[0].shape #具体图像的尺寸
        imge_transform = transforms.ToPILImage()(img[0][0])#读出具体图像
        imge_transform.save("test_reconstruct.jpg")#重建后的图像
        pdb.set_trace()

