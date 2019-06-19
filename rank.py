import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math


def getMatchNum(matches,ratio):
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance:
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

path='./'
queryPath=path+'query_image/' 
samplePath=path+'origin_1.jpg'
comparisonImageList=[] 
sift = cv2.xfeatures2d.SIFT_create() 
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage=cv2.imread(samplePath,0)
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
count=len(comparisonImageList)
column=4
row=math.ceil(count/column)
figure,ax=plt.subplots(row,column)
for index,(image,ratio) in enumerate(comparisonImageList):
    ax[int(index/column)][index%column].set_title('Similiarity %.2f%%' % ratio)
    ax[int(index/column)][index%column].imshow(image)
plt.show()