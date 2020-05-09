from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import argparse
import logging
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
from PIL import Image
from process_image import process_image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image_size = 224

def predict(image_path, saved_model, top_k=1, category_names=None):
    
    # Open and preprocess image
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image, image_size=image_size)
    
    # Load model
    
    model = tf.keras.models.load_model(saved_model, 
                                   custom_objects={
                                       'KerasLayer': tf_hub.KerasLayer
                                       }
                                   )
    
    # Make predictions
    
    prediction = model.predict(image)[0]
    
    if (top_k > 1):
        top_indices = np.argsort(-prediction)[:top_k]
        print('Top {} likely classes:'.format(int(top_k))) 
    elif (top_k == 1):
        top_indices = [np.argmax(prediction)]
        print('Most likely class:')
    else:
        raise TypeError('Please enter a positive integer for k')
    
    # Print results
    
       
    
    if category_names:
        with open('{}'.format(category_names), 'r') as f:
            class_names = json.load(f)
        for x in top_indices:
            print('\nClass: {} with probability {:.3%}'
                  .format(class_names.get(str(x+1)), prediction[x]))
    else:
        for x in top_indices:
            print('\nLabel: {} with probability {:.3%}'
                  .format(x, prediction[x]))

parser = argparse.ArgumentParser(
    description='Predict the class of the output image'
    )

parser.add_argument('image_path', help='Directory path of the image')
parser.add_argument('saved_model', help='Saved pre-trained model')
parser.add_argument('--top_k', action='store', dest='top_k', type=int,
                    help='Returns the top K most likely classes', default=1)
parser.add_argument('--category_names', action='store', dest='category_names',
                    help='Path to a JSON file mapping labels to classes')
args = parser.parse_args()

predict(**vars(args))




    
    
