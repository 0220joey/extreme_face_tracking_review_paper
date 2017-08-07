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
dlib_model = '/home/joey/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
align = openface.AlignDlib(dlib_model)

def get_landmarks(model,img):
    pdb.set_trace()
    bb = model.getLargestFaceBoundingBox(img)
    bb2 = align.getLargestFaceBoundingBox(img)
    landmarks = model.findLandmarks(img,bb)
    return landmarks

if __name__ == "__main__":
    i_images = indoorDF['imgPath']
    i_points = indoorDF['points']
    o_images = outdoorDF['imgPath']
    o_points = outdoorDF['points']
    g_images = globalDF['imgPath']
    g_points = globalDF['points']
    face_model = openface.AlignDlib(dlib_model)
    pdb.set_trace()
    for img,pts in zip(i_images,i_points):
        landmarks = get_landmarks(face_model,img)
        pdb.set_trace()
