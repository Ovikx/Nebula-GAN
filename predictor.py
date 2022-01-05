import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Good generators: 1350 and 1500
generator = tf.keras.models.load_model('models2/generators/nebula_generator_1500.h5')
seeds = tf.random.normal([100, 100])

predictions = (generator(seeds, training=False).numpy()*255).astype('uint8')
for i, prediction in enumerate(predictions):
    print(prediction.shape)
    img = Image.fromarray(prediction)
    img.save(f'test_images/test_{i+1}.jpg')


