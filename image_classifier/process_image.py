import tensorflow as tf
import numpy as np   

def process_image(image, image_size):
    
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255.0
    image = image.numpy()
    image = np.expand_dims(image, axis=0)
    
    return image