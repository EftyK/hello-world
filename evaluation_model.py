import argparse
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


from model import *

BATCH_SIZE = 32 # TODO: is a big batch size good? - research it in google
RANDOM_SEED = 123

damage_intensity_encoding = dict()
damage_intensity_encoding[3] = '3'
damage_intensity_encoding[2] = '2' 
damage_intensity_encoding[1] = '1' 
damage_intensity_encoding[0] = '0' 


###
# Loss function for ordinal loss from https://github.com/JHart96/keras_ordinal_categorical_crossentropy
###
def ordinal_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


parser = argparse.ArgumentParser(description='Test a model')
parser.add_argument('--path', metavar='/path/to/input_model', required=True, help='full path to saved model')
parser.add_argument('--test_dir', metavar='/path/to/test_dir', required=True, help='full path to the directory of test images')
parser.add_argument('--test_csv', metavar='/path/to/test.csv', required=True, help='full path to the test.csv file')


args = parser.parse_args()

model = load_model(args.path, custom_objects={'ordinal_loss': ordinal_loss, 'f1': f1})


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

df = pd.read_csv(args.test_csv)
df = df.replace({"labels" : damage_intensity_encoding })

test_datagen = datagen.flow_from_dataframe(dataframe=df, 
                                        directory=args.test_dir, 
                                        x_col='uuid', 
                                        y_col='labels', 
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        seed=RANDOM_SEED,
                                        class_mode="categorical",
                                        target_size=(128, 128))

predictions = model.predict(test_datagen)

val_trues = test_datagen.classes
val_pred = np.argmax(predictions, axis=-1)

VAL_PRED = np.array(val_pred).tolist()
print("###################### classification_report #######################")
print(classification_report(val_trues, VAL_PRED))
print("###################### confusion_matrix #######################")
print(confusion_matrix(val_trues, VAL_PRED))