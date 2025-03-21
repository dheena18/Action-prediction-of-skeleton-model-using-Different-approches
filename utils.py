import os
from random import randint
from matplotlib import pyplot as plt
import pandas as pd
import csv
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

def mapLabels(label:str):
    if label == "boxing":
        return 0
    elif label == "drums":
        return 1
    elif label == "guitar":
        return 2
    elif label == "rowing":
        return 3
    elif label == "violin":
        return 4

def computeJointDistance(keypoints):

    jointIdx = [(2,5),(3,6), (4,7),(10,13)]
    distance = []

    kp = np.array(keypoints).reshape(-1,3)

    for joint in jointIdx:
        d = computeDistance(kp[joint[0]],kp[joint[1]])
        distance.append(d)
    
    return distance

def computeDistance(joint1, joint2):
    x1 = joint1[0]
    y1 = joint1[1]

    x2 = joint2[0]
    y2 = joint2[1]

    return np.sqrt((x2 - x1)**2 + (y2-y1)**2)
    
def removeConfidenceScore(keypoints):
    # Remove every kth item in-place
    K = 3
    del keypoints[K-1::3] 

    return keypoints   

def removeKeypoints(keypoints):
    removeIdx = [0,1,4,7,8,9,11,12,14,15,16,17,18,19,20,22,23]
    removeIdx.reverse()

    kp = np.array(keypoints).reshape(-1,3)

    for idx in removeIdx:
        kp = np.delete(kp,idx,0)
    
    kp = kp.flatten()

    return kp.tolist()

    
def normalizeKeyPoints(keyPoints):
    poseKeyPoints = np.array(keyPoints).reshape(-1,3)

    # Normalize x and y coordinates
    source_width = 640
    source_height = 480

    poseKeyPoints[:, 0] /= source_width  
    poseKeyPoints[:, 1] /= source_height

    poseKeyPoints = poseKeyPoints.flatten()

    return poseKeyPoints.tolist()



def importData(dirPath:str):
    """
    Returns the dataset as object from the 'dirPath'
    """

    trainSet = []

    trainPath = dirPath + '/train/train/'

    for file in os.listdir(trainPath):
        fileName = os.fsdecode(file)

        # Extract label from file name
        label =  fileName.split("_")[1].split(".")[0]
        # Here we map labeel to int 
        y = mapLabels(label)

        # Create a file object to parse csv files
        file = open(trainPath + fileName,newline='',encoding='utf-8')
        reader = csv.reader(file)
        samples = []
        
        # Extract 'keypoints, 'confidence' and 'angles'
        for row in reader:
            if len(row) == 79:
                keypoints = [float(row[i]) for i in range(0,75)]
                angles = [float(row[j]) for j in range(75,79)]
                poseKeypoints = normalizeKeyPoints(keypoints)
                jointDistance =  computeJointDistance(keypoints)
                keypoints = removeKeypoints(poseKeypoints)
                #Remove confidence score
                poseKeypoints =  removeConfidenceScore(keypoints)
                samples.append(poseKeypoints + angles)

        #Add a sample to 'trainSet'
        features = samples
        trainSet.append((features,y))
    
    return trainSet

def importTestData(dirPath:str):
    """
    """

    testSet = []

    testPath = dirPath + '/test/test/'

    # Sort files
    fileList = sorted(os.listdir(testPath),key=lambda x:int(x.split('.')[0]))

    for file in fileList:
        fileName = os.fsdecode(file)

        # Extract label from file name
        id =  fileName.split(".")[0]

        # Create a file object to parse csv files
        file = open(testPath + fileName,newline='',encoding='utf-8')
        reader = csv.reader(file)
        samples = []
        
        # Extract 'keypoints, confidence' and 'angles'
        for row in reader:
            if len(row) == 79:
                keypoints = [float(row[i]) for i in range(0,75)]
                angles = [float(row[j]) for j in range(75,79)]
                poseKeypoints = normalizeKeyPoints(keypoints)
                jointDistance =  computeJointDistance(poseKeypoints)
                keypoints = removeKeypoints(poseKeypoints)
                #Remove confidence score
                poseKeypoints =  removeConfidenceScore(keypoints)
                samples.append(poseKeypoints + angles)

        #Add a sample to 'trainSet'
        features = samples
        testSet.append((features,id))
    return testSet