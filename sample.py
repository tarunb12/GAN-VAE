# %% Run from root
import os
if os.getcwd() == os.path.dirname(os.path.abspath(__file__)):
    os.chdir('..')

# %% Import ML libraries
# pylint: disable=E0401
from config import ROOT_DIR
import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% Constants
EPOCHS = 10
NUM_CLASSES = 10

# %% Set up training and testing data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
image_length, image_width = train_images.shape[1], train_images.shape[2]
train_images, train_labels = train_images / 255.0, train_labels
test_images, test_labels = test_images / 255.0, test_labels

# %% Create model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(image_length, image_width)),
    # keras.layers.Dense(units=512, activation=keras.activations.relu),
    keras.layers.Dense(units=256, activation=keras.activations.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(NUM_CLASSES),
    keras.layers.Softmax(),
])
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        keras.metrics.SparseCategoricalAccuracy()
    ]
)
model.summary()

# %% Set up progress saving by creating a checkpoint callback
checkpoint_path = 'tmp/training.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
)

# %% Train the model and save
history = model.fit(
    x=train_images,
    y=train_labels,
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
    callbacks=[checkpoint_callback],
)

model_dir = os.path.join(ROOT_DIR, 'models', 'model')
model.save(model_dir)

# %% Import plotting libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# %% Save plots
analysis_path = os.path.join(ROOT_DIR, 'analysis', 'model')

train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

train_loss = history.history['loss']
test_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Testing Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.savefig(os.path.join(analysis_path, 'loss_and_accuracy.png'))
plt.close()

test_preds = model.predict(test_images)
predictions = tf.argmax(test_preds, axis=-1)
confusion_matrix = tf.math.confusion_matrix(test_labels, predictions, num_classes=NUM_CLASSES).numpy()

plt.figure(figsize=(20, 14))
df_confusion_matrix = pd.DataFrame(confusion_matrix, range(10), range(10))
sn.set(font_scale=1.4)
sn.heatmap(df_confusion_matrix, annot=True, annot_kws={'size': 16}, fmt='d', cmap='rocket_r')
plt.savefig(os.path.join(analysis_path, 'confusion_matrix.png'))
plt.close()
