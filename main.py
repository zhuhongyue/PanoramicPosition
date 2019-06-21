# -*- coding: utf-8 -*
import cv2
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import scipy.misc
from scipy.spatial.distance import pdist
import pdb
from panoramicDataset import panoramicDataset
from matchRatioRank import matchRatioRank


if __name__ == "__main__":
    print "matching through dataloader method"
    troch_trainset = panoramicDataset.trainset('/home/zjfan/SmartCityCompaign/dataset/train_data/scene1_jiading_lib_training/scene1_jiading_lib_training_coordinates.csv')
    trainloader = panoramicDataset.DataLoader(troch_trainset, batch_size=1,shuffle=False)#do not test other batchsize
    indexImg = cv2.imread("/home/zjfan/SmartCityCompaign/PanoramicPosition/test_reconstruct.jpg",0)
    totalSceneList = matchRatioRank.imageRank_SIFT(indexImg,trainloader)
    pdb.set_trace()