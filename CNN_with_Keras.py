'''
A convolution neural network experiment

Trains a network to recognize if the given picture has a Dog or a Cat

Dataset = Microsoft Kaggle
'''

import tensorflow as tf
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

create_data = False  #True if you want to normalize the data and save them in a numpy array
post_process = False # set True if you want to dump the data from the numpy array into a pickle


location = "E:\\road to ML\\MNIST_Number_Classifier\\PetImages"
IMG_size = 100
labels = ["Cat","Dog"]
def create_training_data():
    train_set = []
    for label in tqdm(labels):
        path = os.path.join(location,label)
        class_idx = labels.index(label)
        for file in tqdm(os.listdir(path)):
            try:
                img = os.path.join(path,file)
                img_array = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img_array,(IMG_size,IMG_size))
                train_set.append([new_img,class_idx])
            except Exception as e:
                pass
    np.save("Training_Data",train_set)

def data_preparation():
    train_set = np.load("Training_Data.npy",allow_pickle=True)
    random.shuffle(train_set)

    x = []
    y = []

    for images,labels in train_set:
        x.append(images)
        y.append(labels)


    x_in = np.array(x).reshape(-1,IMG_size,IMG_size,1)

    pickle_out = open("x_pickle","wb")
    pickle.dump(x_in,pickle_out)
    pickle_out.close()
    print(" x has been saved")

    pickle_out = open("y_pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    print(" y has been saved")

if create_data  : create_training_data()
if post_process : data_preparation()
print("Beginning Training")

# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

Name= "Cats_VS_Dogs_64X2_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs\{}'.format(Name))

pickle_in = open("x_pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
X= np.array(X)
y= np.array(y)
model= Sequential()
model.add(Conv2D(64,(3,3),input_shape= X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss= "binary_crossentropy",optimizer = "adam"
                , metrics=["accuracy"])
model.fit(X,y,batch_size=32,epochs=10, validation_split = 0.2,callbacks =[tensorboard])

'''
Work to be done

=> Optimize the model by varying the number of hidden layers

'''