# IMPORT #
# 1. DL libraries (tf, keras, layers, preprocessing, etc...)
# 2. Viz libraries (matplotlib, PIL, etc...)
# 3. File handling libraries (os, csv, json, pathlib, etc...)

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

# DATASET #
# retrieve data
# split them

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)


# CHECK #
# file system structure
# file size
# viz

print("\n-------METADATA-------\n")
print(metadata)

num_classes = metadata.features['label'].num_classes
print("\n-----NUMBER OF CLASSES-------\n")
print(num_classes)

get_label_name = metadata.features['label'].int2str

print("\n-------DATASETS CARDINALITY---------\n")
print(train_ds.cardinality().numpy())
print(val_ds.cardinality().numpy())
print(test_ds.cardinality().numpy())

print("\n-----IMAGE SHAPE---------\n")
iterator = iter(train_ds)
for i in range(4):
    image, label = iterator.get_next()
    plt.subplot(2, 2, i+1)
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.axis("off")
    print(image.shape)

plt.show()
plt.close()

# RESCALING AND DATA AUGMENTATION #
AUTOTUNE = tf.data.experimental.AUTOTUNE

batch_size = 32
image_size = (128, 128)

resize = keras.Sequential([keras.layers.experimental.preprocessing.Resizing(image_size[0], image_size[1])], name="resize")

train_ds = train_ds.map(lambda x, y: (resize(x), y)).batch(batch_size).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (resize(x), y)).batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

rescale = keras.Sequential([
    keras.layers.experimental.preprocessing.Rescaling(1. / 255)
], name="rescale")

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomContrast(.2, seed=123),
    keras.layers.experimental.preprocessing.RandomFlip(),
    keras.layers.experimental.preprocessing.RandomRotation(.5),
    keras.layers.experimental.preprocessing.RandomZoom(.3, .3)
], name="data_aug")

# MODEL STRUCTURE #

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(*image_size, 3)),
    rescale,
    data_augmentation,
    keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    keras.layers.Dropout(.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_classes)
])

model.summary()

model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

epochs = 15
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# PLOT RESULTS #

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# PREDICT #
print("\n------PREDICT---------\n")
test_ds = test_ds.map(lambda x, y: (resize(x), y)).batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
test_loss, test_acc = model.evaluate(test_ds)

print("\nTest loss: {}".format(test_loss))
print("\nTest accuracy: {}".format(test_loss))





