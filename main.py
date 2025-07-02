import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import app



# Training data generator: augments and normalizes
train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=15,
  zoom_range=0.1,
  horizontal_flip=True
)

# Validation data generator: only normalize 
val_datagen = ImageDataGenerator(rescale=1./255)

# load training images 
train_img = train_datagen.flow_from_directory(
  'Covid19-dataset/train', # path
  target_size=(224,224), # resie all images
  batch_size=32,
  class_mode='sparse', # multiclass classification
  color_mode = 'grayscale'
)

# Load validation images 
val_img = val_datagen.flow_from_directory(
  'Covid19-dataset/test',
  target_size=(224, 224),
  batch_size=32,
  class_mode='sparse',
  color_mode='grayscale'
)

# Create a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(224,224,1))) # input layer
model.add(Flatten()) # flatten layer in between
model.add(Dense(3, activation='softmax')) # output layer 

# Model compile with Adam optimizer
model.compile(
  optimizer=Adam(learning_rate=0.001),
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.summary()

history = model.fit(
  train_img,
  epochs=20,
  validation_data=val_img,
  verbose=1
)

# Do Matplotlib extension below
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping
fig.tight_layout()

fig.savefig('static/images/my_plots.png')
