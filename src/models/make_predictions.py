import os

import tensorflow as tf
from tensorflow import keras

from preprocess_data import TextCleanTransformer

cleaner = TextCleanTransformer()
text = "Forest fire near La Ronge Sask. Canada"
clean_text = cleaner.preprocess_sentence(text)
model = tf.keras.models.load_model('models/my_model')
predicted = model.predict([text])

print(predicted)
