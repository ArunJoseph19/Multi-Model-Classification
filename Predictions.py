# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 09:17:32 2020

@author: A ARUN JOSEPHRAJ
"""
# Prediction of different images

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing import image

def show_images(data):
        plt.axis('off')
        plt.imshow(data)
        plt.show()

Main_Model = keras.models.load_model('models/Main')

#predicting a single image
folder ='dataset/test_set/Kitten V Ice Cream/1 (1).jpg'

img = image.load_img(folder, target_size = (250,250))
test_image = image.load_img(folder, target_size = (250,250))
test_image = image.img_to_array(test_image) #converting it into an array
test_image = np.expand_dims(test_image, axis = 0) 
result = Main_Model.predict(test_image)
result = np.round(result)

if result[0][0] == 1:
    Label = "Chihuahua V Muffin"
    Sub_Model = keras.models.load_model('models/ChihuahuaVMuffin')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Chihuahua"
    else:
        title = "Muffin"
elif result[0][1] == 1:
    Label = "Kitten V Ice Cream"
    Sub_Model = keras.models.load_model('models/KittenVIceCream')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Kitten"
    else:
        title = "Ice Cream"
elif result[0][2] == 1:
    Label = "Labradoodle V Fried Chicken"
    Sub_Model = keras.models.load_model('models/LabradoodleVFriedChicken')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Labradoodle"
    else:
        title = "Fried Chicken"
elif result[0][3] == 1:
    Label = "Parrot V Guacamole"
    Sub_Model = keras.models.load_model('models/ParrotVGuacamole')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Parrot"
    else:
        title = "Guacamole"
elif result[0][4] == 1:
    Label = "Puppy V Bagel"
    Sub_Model = keras.models.load_model('models/PuppyVBagel')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Puppy"
    else:
        title = "Bagel"
elif result[0][5] == 1:
    Label = "Sheepdog V Mop"
    Sub_Model = keras.models.load_model('models/SheepdogVMop')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Sheepdog"
    else:
        title = "Mop"
elif result[0][6] == 1:
    Label = "Shiva Inu V Marshmallow"
    Sub_Model = keras.models.load_model('models/ShivaInuVMarshmallow')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Shiva Inu"
    else:
        title = "Marshmallow"
else:
    Label = "Sloth V Croissant"
    Sub_Model = keras.models.load_model('models/SlothVCroissant')
    prediction = Sub_Model(test_image)
    if prediction[0][0] == 1:
        title = "Sloth"
    else:
        title = "Croissant"
    
show_images(img)
print("The image belongs to:",Label,"\nThe image is of a: ",title)
