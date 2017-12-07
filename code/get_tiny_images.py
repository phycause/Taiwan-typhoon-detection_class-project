from PIL import Image

import pdb
import numpy as np
from sklearn import preprocessing

def get_tiny_images(image_paths):

    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    先將影像切成正方形的，再進行resize
    '''
    #目前最好 new_width = new_height = 20, new_length = 345
    new_width = 20
    new_height = 20
    tiny_images_ori = []

    for img_file in image_paths:
        img = Image.open(img_file)
        #取得圖片的大小
        width, height = img.size
        
        #計算切割的大小(以最小損失切割成正方形)
        
        new_length = 345
        #計算要切的範圍(以圖片中心為切割中心)
        left = (width - new_length)/2
        top = (height - new_length)/2
        right = (width + new_length)/2
        bottom = (height + new_length)/2
        img_cropped = img.crop((left, top, right, bottom))
        
        #resize圖片大小
        img_resized = img_cropped.resize((new_width, new_height), Image.BILINEAR)
        #將調整過後的圖片轉成float32存入tiny_images
        tiny_images_ori.append(np.asarray(img_resized,dtype='float32').flatten().tolist())
    
    tiny_images = np.array(tiny_images_ori)
    #將feature做mean unit length normalization
    tiny_images_T = np.transpose(tiny_images)

    for idx, feature in enumerate(tiny_images_T):
        feature_scaled = preprocessing.scale(feature,with_std = True)
        # feature_scaled = feature_scaled/np.max(abs(feature_scaled))
        tiny_images[:,idx] = feature_scaled

    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
