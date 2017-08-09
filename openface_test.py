import openface
import numpy as np
import cv2
import dlib
import pandas
import pdb
from PIL import ImageEnhance, Image

data_dir = '/media/drive/ibug/300W/'

indoor = data_dir + '01_Indoor/'
outdoor = data_dir + '02_Outdoor/'
dlib_model = '/home/joey/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
if __name__ == "__main__":
    align = openface.AlignDlib(dlib_model)
    imgPath = indoor + 'indoor_112.png'
    rgbImg = cv2.imread(imgPath)
    rgbImg = cv2.cvtColor(rgbImg,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgbImg)
    enhancer = ImageEnhance.Brightness(pil_im)
    enhancer.enhance(255).show()
    '''
    alpha = float(1)
    beta = float(255)
    mul_img = cv2.multiply(rgbImg,np.array([alpha]))
    new_img = cv2.add(mul_img,np.array([beta]))

    cv2.imshow('original_image', rgbImg)
    cv2.imshow('new_image',new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if rgbImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    landmarks = align.findLandmarks(rgbImg,bb)
    for (x, y) in landmarks:
        cv2.circle(rgbImg, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Output", rgbImg)
    cv2.imwrite('rgb_landmarks.png',rgbImg)
    pdb.set_trace()
    '''
