
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.optimizers import SGD
import cv2, numpy as np
from keras import backend as K
import config
import numpy as np
from sklearn.cross_validation import train_test_split
import config
import h5py
""" Get Data """

import scipy.io as sio
from datetime import date
a=date.fromordinal(723671).year

wiki= sio.loadmat('wiki.mat')
age=[]
day=wiki["wiki"][0][0][0][0]
photo_taken_year=wiki["wiki"][0][0][1][0]

birth=[]
for i in day:
    birth.append(date.fromordinal(i).year)
print len(birth),len(photo_taken_year)
age_list=[]

name= wiki["wiki"][0][0][2][0]
for i in range(len(birth)):
    age_list.append((photo_taken_year[i]-birth[i]))
new_age_list=[]
new_name=[]
for i in range(len(birth)):
    if age_list[i]>10 and age_list[i]<100:
        new_age_list.append(age_list[i])
        new_name.append(name[i])
name_path=[]
for i in new_name:
    name_path.append(i[0])

data normalization
new=[]
for i in new_age_list[0:100]:
    new.append(i)
real_list=[]
for i in new_age_list[0:100]:
    a=float(i-min(new))/(max(new)-min(new))
    real_list.append(a)

def VGG_16(weights_path=None):
    model = Sequential()
    K.set_image_dim_ordering('th')

    
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Conv2D(64,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,( 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,( 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,( 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,( 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,( 3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,( 3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(512,( 3, 3)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    img_array=[]
    for i in name_path[0:100]:
        im = cv2.resize(cv2.imread(i), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        im=im.reshape(3,224,224)
        img_array.append(im)
    img_array=np.array(img_array)

    model = VGG_16('my_model.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')

    # out = model.predict(im)
    # print np.argmax(out)
    #%%
    (X, y) = (img_array[0:100],real_list[0:100])
    # normalize data
    X=np.array(X)
    y=np.asarray(y)
    print y
    y=y.reshape(len(y),1)

    train_data,test_data, train_label, test_label= train_test_split(X, y, test_size=0.5, random_state=4)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    """ Normalize Data """
    train_data = train_data / 255
    test_data = test_data / 255

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    previous_best_loss=100000000
    for i in range(config.epoch):
        print train_data.shape,train_label.shape
        model.fit(train_data, train_label,batch_size=config.batch_size,
            epochs=int(config.epoch/config.epoch),
            validation_data=(test_data, test_label),
            shuffle=True)
        train_record = model.evaluate(train_data, train_label)
        print 'train_Record',(train_record)
        train_loss_list.append(train_record)
        val_record = model.evaluate(test_data, test_label)
        val_loss_list.append(val_record)
        current_loss=train_record
        if current_loss < previous_best_loss:
            model.save('my_model.h5')
            previous_best_loss=current_loss

#     ++++++++++++++++++print("\nAverage testing accuracy: {}%".format(round(sum(val_loss_list)/len(val_loss_list), 4) * 100))
#     ++++++++++++print("Maximum testing lost: {}%".format(round(max(val_loss_list), 4) * 100))
#     ++++++print("Last epoch testing lost: {}%".format(round(val_loss_list[-1], 4) * 100))
#     #save model
    


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
class_names=[i for i in range(3)]
pred_label= model.predict(test_data)
print pred_label
# for i in pred_label:
#     i=i*(max(new)-min(new))+min(new)
print pred_label
pred_label=list(itertools.chain.from_iterable(pred_label))
test_label=list(itertools.chain.from_iterable(test_label))
print pred_label
print test_label


# In[13]:


from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(test_label, pred_label)
# np.set_printoptions(prescision=2)
plot.plot_confusion_matrix(cnf_matrix, class_names, False, "Confusion matrix", "/Users/nicole/Desktop/python/finalproject")
plot.plot_curve(epoch_list, train_loss_list, "Accuracy curve", "/Users/wei-jer-chang/Desktop/final project", training=True, accuracy=False)
   

