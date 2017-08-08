import cv2
import openface
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb

data_dir = '/media/drive/ibug/300W/'
indoor = data_dir + '01_Indoor/'
outdoor = data_dir + '02_Outdoor/'
indoorDF =  pd.read_csv('indoor_300w.csv')
outdoorDF = pd.read_csv('outdoor_300w.csv')
globalDF = pd.read_csv('global_300w.csv')
dlib_model ='/Users/joeybose/openface/models/dlib/shape_predictor_68_face_landmarks.dat'

def get_landmarks(model,rgbimg):
    bb = model.getLargestFaceBoundingBox(rgbimg)
    landmarks = model.findLandmarks(rgbimg,bb)
    return np.asarray(landmarks)

def get_avg_dist(pt_labels,landmarks,norm):
    total_dist = 0
    pdb.set_trace()
    for pt,pred in zip(pts,landmarks):
        dist = np.linalg.norm(pt - pred)
        total_dist = total_dist + dist
    return np.divide(total_dist,norm)

if __name__ == "__main__":
    i_images = indoorDF['imgPath']
    i_points = indoorDF['points']
    o_images = outdoorDF['imgPath']
    o_points = outdoorDF['points']
    g_images = globalDF['imgPath']
    g_points = globalDF['points']
    face_model = openface.AlignDlib(dlib_model)
    for img,pts in zip(i_images,i_points):
        rgbImg = cv2.imread(img)
        landmarks = get_landmarks(face_model,rgbImg)
        left_inner = landmarks[39]
        right_inner = landmarks[42]
        inter_occ_dist = np.linalg.norm(left_inner - right_inner)
        img_dist = get_avg_dist(pts,landmarks,inter_occ_dist)
