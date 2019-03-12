
# coding: utf-8

# In[4]:

import cv2
'''all height, length must be ODD!!'''

#API
#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
'''
def Convolu(img, height, length):
    kernel = np.ones((height,length),np.float32)/(height*length)
    img = cv2.filter2D(img,-1,kernel)
    return img
'''        
#Average and Gaussian
#https://mmeysenburg.github.io/image-processing/05-blurring/

def Average(img, height, length):
    img = cv2.blur(img,(height, length))
    return img
#replace centre pixel with the average value of all the elements in the metrix

def Gaussian(img, height, length, devi):
    img = cv2.GaussianBlur(img,(height, length),devi)
    return img
#replace centre pixel with the average value of all the elements in the metrix, cosidering weight

def Median(img, x):
    img = cv2.medianBlur(img,x)
    return img
#https://www.cs.auckland.ac.nz/courses/.../Image%20Filtering_2up.pdf

#x means how many pixel you want to copare at one time. 
#[2,3,4,31,1,23,43,3]
#x= 3
#[2,3,4]-->sort-->pick the middle-->3
#[3,4,31]-->sort-->pick the middle-->4
#[4,31,1]-->sort-->pick the middle-->4
def Bilateral(img, diameter, sigmaColor, sigmaspace):
    img = cv2.bilateralFilter(img,diameter, height, length)   
    return img
#parameter
#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
#intro
#https://cg2010studio.com/2012/10/14/%E9%9B%99%E9%82%8A%E6%BF%BE%E6%B3%A2%E5%99%A8-bilateral-filter/

#pixels lying at the edges displaying large intensity variations 
#will not be included in the blurring and hence be preserved.


def Rotate(img, degree):
    rows,cols,ch = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    #(center pos, degree, CCW)
    img = cv2.warpAffine(img,M,(cols,rows))
    return img
#affine trasformation
#INTRO: http://monkeycoding.com/?p=605#more-605


# In[12]:

import numpy as np
a = np.array([[[1,2,3] ,[3,21,3],[1,1,2]],[[1,2,3] ,[3,21,3],[1,1,2]]])
print a
#a = a.transpose((2,0,1))
#print a
a = np.expand_dims(a, axis=0)
print a
b = a.reshape(2,3,3)
print b-a


# In[ ]:



