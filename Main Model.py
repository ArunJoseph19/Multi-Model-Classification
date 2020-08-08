# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 08:48:51 2020

@author: A ARUN JOSEPHRAJ
"""

# Convolutional Neural Network
# For the Main model

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255, #feature scaling
                                   shear_range = 0.2, 
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Dataset/training_set', #path to folder
                                                 target_size = (250, 250), #size of the images   
                                                 batch_size = 2,        
                                                 class_mode = 'categorical')

len(training_set)
# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[250, 250, 3]))
#filters : no of parameters
#kernel size: matrix size of feature detector
#input_shape: color images and we resize so (64,64,3)

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
#gets maximum value within a 2x2 matrix for each....that's pool_size
#strides: how much it moves to next.......it moves 2-2 steps

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())
#converting it into a 1-D array is flattening

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=8, activation='softmax'))

cnn.summary()

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, epochs = 15)

#saving the model
cnn.save('Models/Main')


