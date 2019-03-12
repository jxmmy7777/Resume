# coding: utf-8

# In[9]:


import sys
from PIL import Image
import numpy as np
def argument_parser(argv):  
    imagename = sys.argv[1]
    saveimagename = sys.argv[2]
    filtertype = sys.argv[3]
    read = read_image(imagename)
    img= image_to_list(read)
    result = []
    w,h=read.size
    
    if len(sys.argv)>=4:
        if filtertype == 'HPF':
            threshold = int(sys.argv[4])
            result=high_pass_filter(img, threshold,w,h)
        elif filtertype == 'LPF':
            threshold = int(sys.argv[4])
            result=low_pass_filter(img, threshold,w,h)
        elif filtertype == 'BPF':
            low_threshold= int(sys.argv[4])
            high_threshold= int(sys.argv[5])
            result=band_pass_filter(img, low_threshold, high_threshold,w,h)
        img = list_to_image(result)
        save_image(img,saveimagename)
    
    


def read_image(filename):

    img = Image.open(filename)
    return img


def image_to_list(img):

    image_array=np.array(img)
    
    return image_array


def high_pass_filter(image_array, threshold,w,h):
    for y in range(h):
        for x in range(w):
            for i in range(3):
                if image_array[y][x][i]<threshold:
                    image_array[y][x][i]=0
    return image_array


def band_pass_filter(image_array, low_threshold, high_threshold,w,h):
    for y in range(h):
        for x in range(w):
            for i in range(3):
                if image_array[y][x][i]<low_threshold or high_threshold<image_array[y][x][i]:
                    image_array[y][x][i]=0

    return image_array


def low_pass_filter(image_array, threshold,w,h):
    for y in range(h):
        for x in range(w):
            for i in range(3):
                if image_array[y][x][i]>threshold:
                    image_array[y][x][i]=0

    return image_array


def list_to_image(image_array):
    array_to_image = Image.fromarray(image_array.astype('uint8'),'RGB')
    return array_to_image


def save_image(img, saveimagename):

    img.save(saveimagename)
    

if __name__ == '__main__':
    argument_parser(sys.argv)


# In[ ]:


