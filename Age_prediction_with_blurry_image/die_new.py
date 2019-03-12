from keras.backend.tensorflow_backend import set_session

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Activation, Lambda
from keras.optimizers import SGD, Adam
import cv2, numpy as np
from keras import backend as K
import config, select
import numpy as np
from sklearn.cross_validation import train_test_split
import config
import h5py
from keras.utils import np_utils
from blur_func import *
from keras.utils import np_utils

""" Get Data """

import scipy.io as sio
from datetime import date
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

a=date.fromordinal(723671).year

wiki= sio.loadmat('wiki.mat')
age=[]
day=wiki["wiki"][0][0][0][0]
photo_taken_year=wiki["wiki"][0][0][1][0]

birth=[]
for i in day:
    birth.append(date.fromordinal(i).year)
age_list=[]

name= wiki["wiki"][0][0][2][0]
for i in range(len(birth)):
    age_list.append((photo_taken_year[i]-birth[i]))
new_age_list=[]
new_name=[]
for i in range(len(birth)):
    if age_list[i]>0 and age_list[i]<=100:
        new_age_list.append(age_list[i])
        new_name.append(name[i])
name_path=[]
for i in new_name:
    name_path.append(i[0])

new, name_path = select.select(new_age_list, name_path)    

    
''''''''''''''''''''''''''''''''''''
def VGG16(weights_path=None):
    
    model = Sequential()
    K.set_image_dim_ordering('th')
    model.add(ZeroPadding2D((1,1),input_shape=(3, 224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((3,3)))
    
    model.add(Flatten())
    model.add(Dense(4090, activation='relu'))
    model.add(Dense(4090, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))
    model.summary()
    if weights_path:
            model.load_weights(weights_path)

    return model

     
''''''''''''''''''''''''''''''''''''
if __name__ == "__main__":
    img_array=[]
    for i in name_path[0:8500]:
        im = cv2.resize(cv2.imread(i), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        im=im.reshape(3,224,224)
        img_array.append(im)
        
        
    model = VGG16('modelWith128and100epoch.h5')
    epochs = config.epoch
    img_array=np.array(img_array)
    (X, y) = (img_array[0:8500],new[0:8500])
    learning_rate = 0.1/(epochs)
    
    
    # normalize data
    X=np.array(X)
    y = np_utils.to_categorical(y, num_classes=100)
    train_data,test_data, train_label, test_label= train_test_split(X, y, test_size=0.2, random_state=4)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data = train_data / 255
    test_data = test_data / 255

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    data = [float(number)
        for line in open('clear_loss.csv', 'r')
        for number in line.split()]
    for x in data:
        data = float(x)
    previous_best_loss=data

    
    '''Here'''
    for i in range(config.epoch):
        print ("epoch", i)
        #print train_data.shape,train_label.shape
        
        if (i+1)%20==0:
            learning_rate=learning_rate/2
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        #adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=sgd, loss='mean_squared_error')
        model.fit(train_data, train_label,batch_size=config.batch_size,
            epochs=int(config.epoch/config.epoch),verbose=1
            ,validation_data=(test_data, test_label),
            shuffle=True)
        print 'predict',model.predict(train_data)
        print 'label',train_label
        train_record = model.evaluate(train_data, train_label)
        
        #print 'train_Record',(train_record)
        
        train_loss_list.append(train_record)
        val_record = model.evaluate(test_data, test_label)
        val_loss_list.append(val_record)
        current_loss=train_record   
        
        if current_loss <= previous_best_loss:
            model.save('die_model.h5')
            #model.save_wights("Median_weight.h5")
            previous_best_loss=current_loss
            #filename = "clear_loss"+str(i)+".csv"
            with open("clear_loss.csv","w") as f1:
                ans=str(previous_best_loss)
                f1.write(ans)
            

print("\nAverage testing lost: {}%".format(round(sum(val_loss_list)/len(val_loss_list), 4) * 100))
print("Maximum testing lost: {}%".format(round(max(val_loss_list), 4) * 100))
print("Last epoch testing lost: {}%".format(round(val_loss_list[-1], 4) * 100))

    



# In[12]:


import plot
import itertools
import numpy as np
import matplotlib
""" Save figure without displaying it """
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pylab import rcParams

epoch_list=[i for i in range(config.epoch)]
pred_label= model.predict(test_data)
for i in pred_label:
     i=i*(max(new)-min(new))+min(new)

"Flatten the label"
pred_label=list(itertools.chain.from_iterable(pred_label))
test_label=list(itertools.chain.from_iterable(test_label))


"Classify the label"
from keras import classification
pred_list=classification.classification(pred_label)
test_list=classification.classification(test_label)
# print pred_list
# print test_list


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(test_list, pred_list)
class_names=["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]
plot.plot_confusion_matrix(cnf_matrix, class_names, False, "Confusion matrix", "")
plot.plot_curve(epoch_list, train_loss_list, "Clear Accuracy curve", "", training=True, accuracy=False)
   

"write file"
import csv
all_value=[[i for i in range(1,config.epoch+1)],train_loss_list,val_loss_list]
title='epoch'+","+'train_loss_list'+","+'val_loss_list'
row_value=zip(*all_value)
sav_value=title+"\n"
data=sav_value
for i in range(len(row_value)):
    data+=str(row_value[i][0])+","+str(row_value[i][1])+"\n"

with open("clear_data.csv", "w") as f1:
    f1.write(data) 
config2 = tf.ConfigProto()
config2.gpu_options.allocator_type ='BFC'
config2.gpu_options.per_process_gpu_memory_fraction = 1.0

