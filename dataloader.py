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
preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
def default_loader(path):
    dirs = os.listdir(path)
    img = []
    for i in dirs:                             
        if os.path.splitext(i)[1] == ".jpg" and "origin" in i:   
            img.append(preprocess(Image.open(path+"/"+i)))
            #img_pil = img_pil.resize((224,224))
            #img_tensor = preprocess(img_pil)
    return img

class trainset(Dataset):
    def __init__(self, path):
        self.images, self.target = readData(path,0)
        self.loader = default_loader
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    img_set, loc = readData('/home/zjfan/SIFT_solution/dataset/train_data/scene1_jiading_lib_training/scene1_jiading_lib_training_coordinates.csv',0)
    troch_trainset = trainset('/home/zjfan/SIFT_solution/dataset/train_data/scene1_jiading_lib_training/scene1_jiading_lib_training_coordinates.csv')
    #load image from certain path
    for i in img_set:
        print i
    trainloader = DataLoader(troch_trainset, batch_size=4,shuffle=True)
    #example for loading data
    for image, location in trainloader:
        print "image len:", len(image)
        print "image shape:", image[0].shape
        print "location:",location
    pdb.set_trace()