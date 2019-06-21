# -*- coding: utf-8 -*
import cv2
import matplotlib as mpl
from torchvision import transforms, utils
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import scipy.misc
from scipy.spatial.distance import pdist
import pdb

def getMatchNum(matches,ratio):
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance:
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

def EuclideanDistance(x,y):
    X=np.vstack([x,y])
    d2=pdist(X)
    return d2

def ManhattanDistance(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'cityblock')
    return d2

def ChebyshevDistance(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'chebyshev')
    return d2

def MinkowskiDistance(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'minkowski',p=2)
    return d2

def imageRank_tensor(image,query_dataloader,distance = EuclideanDistance):
    #TODO:using deep network to get match ratio sorted list

    return

def imageRank_SIFT(sampleImage,query_dataloader):
    sift = cv2.xfeatures2d.SIFT_create() 
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams=dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)
    kp1, des1 = sift.detectAndCompute(sampleImage, None) 
    totalSceneList = []
    for __, img, location, fn in query_dataloader:
        comparisonImageList = []
        matchRatioSum = 0
        for (img_instance,fn_instance) in zip(img,fn):
            queryImage = img_instance[0]
            queryImage = transforms.ToPILImage()(queryImage)
            queryImage = np.array(queryImage)
            kp2, des2 = sift.detectAndCompute(queryImage, None)
            matches=flann.knnMatch(des1,des2,k=2) 
            (matchNum,matchesMask)=getMatchNum(matches,0.9) 
            matchRatio=matchNum*100/len(matches)
            matchRatioSum+=matchRatio
            comparisonImageList.append((fn_instance,matchRatio)) 
        comparisonImageList.sort(key=lambda x:x[1],reverse=True) 
        totalSceneList.append((comparisonImageList,location,matchRatioSum,os.path.split(fn)[0]))
    totalSceneList.sort(key=lambda x:x[2],reverse=True) 
    pdb.set_trace()
    return totalSceneList


def imageRank_path(path = './',queryPath = 'query_image/',samplePath = '6987.jpg'):
    queryPath=path + queryPath 
    samplePath=path + samplePath
    comparisonImageList=[] 
    sift = cv2.xfeatures2d.SIFT_create() 
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams=dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)
    sampleImage=cv2.imread(samplePath,0)
    pdb.set_trace()
    kp1, des1 = sift.detectAndCompute(sampleImage, None) 
    for parent,dirnames,filenames in os.walk(queryPath):
        for p in filenames:
            p=queryPath+p
            queryImage=cv2.imread(p,0)
            kp2, des2 = sift.detectAndCompute(queryImage, None) 
            matches=flann.knnMatch(des1,des2,k=2) 
            (matchNum,matchesMask)=getMatchNum(matches,0.9) 
            matchRatio=matchNum*100/len(matches)
            drawParams=dict(matchColor=(0,255,0),
                    singlePointColor=(255,0,0),
                    matchesMask=matchesMask,
                    flags=0)
            comparisonImage=cv2.drawMatchesKnn(sampleImage,kp1,queryImage,kp2,matches,None,**drawParams)
            comparisonImageList.append((comparisonImage,matchRatio)) 
    comparisonImageList.sort(key=lambda x:x[1],reverse=True) 
    return comparisonImageList
    
if __name__ == "__main__":
    print "matching through file path method"
    comparisonImageList = imageRank_path(path = './',queryPath = 'query_image/',samplePath = '6987.jpg')
    print "rank list:"
    for index,(image,ratio) in enumerate(comparisonImageList):
        print "Match ratio:",ratio
        scipy.misc.imsave('{}.jpg'.format(index), image)





