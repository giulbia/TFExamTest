import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

# DATASET #
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

data_path = keras.utils.get_file("aclImdb_v1.tar.gz", origin=url, untar=True)

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'imdb_reviews',
    split=['train', 'test', 'unsupervised'],
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

print("\n-----TRAINING TEXT EXAMPLES---------\n")
iterator = iter(train_ds)
for i in range(5):
    text, label = iterator.get_next()
    review_length = tf.strings.length(text).numpy()
    print("({}) - {}: {}".format(review_length, get_label_name(label), text))

print("\n-----TEST TEXT EXAMPLES---------\n")
iterator = iter(test_ds)
for i in range(5):
    text, label = iterator.get_next()
    review_length = tf.strings.length(text).numpy()
    print("({}) - {}: {}".format(review_length, get_label_name(label), text))

print("\n-------CLASS BALANCE--------\n")

iterator = iter(train_ds)
nb_pos = 0
for element in iterator:
    nb_pos += element[-1].numpy()

print("Positive examples in training set: {}".format(nb_pos))

iterator = iter(val_ds)
nb_pos = 0
for element in iterator:
    nb_pos += element[-1].numpy()

print("Positive examples in validation set: {}".format(nb_pos))

iterator = iter(test_ds)
# print(iterator.get_next())
nb_pos = 0
for element in iterator:
    nb_pos += element[-1].numpy()

print("Positive examples in test set: {}".format(nb_pos))
print("3rd class is unsupervised")

# PREP DATA #
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 512

max_features = 10000  # Maximum vocab size.
max_len = 500  # Sequence length to pad the outputs to
embedding_dim = 16

vectorize_layer = keras.layers.experimental.preprocessing.TextVectorization(
    standardize="lower_and_strip_punctuation",  # default
    split="whitespace",  # default
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=max_len,
    pad_to_max_tokens=True  # default
)

# Make a text-only dataset (without labels), then call adapt
train_text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text_ds)

print("\n--------PREP DATA-------\n")
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print("Vocabulary size: {}".format(len(vectorize_layer.get_vocabulary())))

iterator = iter(train_text_ds)
for i in range(1):
    text = iterator.get_next()
    text = tf.expand_dims(text, -1)
    v_text = vectorize_layer(text)
    print("{} -> {}".format(text, v_text))


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = train_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE).map(vectorize_text)
val_ds = val_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE).map(vectorize_text)

# MODEL #

# model = tf.keras.Sequential([
#     keras.layers.Embedding(max_features + 1, embedding_dim, mask_zero=True),
#     # keras.layers.SimpleRNN(32, return_sequences=True), OVERFIT À fond
#     keras.layers.SimpleRNN(32),
#     keras.layers.Dense(1)])

model = tf.keras.Sequential([
    keras.layers.Embedding(max_features + 1, embedding_dim, mask_zero=True),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)])

model.summary()

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='rmsprop',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[tensorboard_callback])


acc = history.history["binary_accuracy"]
val_acc = history.history["val_binary_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.plot(range(epochs), acc, label="training")
plt.plot(range(epochs), val_acc, label="validation")
plt.title("Accuracy")
plt.legend(loc="lower right")
plt.subplot(122)
plt.plot(range(epochs), loss, label="training")
plt.plot(range(epochs), val_loss, label="validation")
plt.title("Loss")
plt.legend(loc="upper right")
plt.show()

# TEST EVALUATION #
loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# PREDICTION #
transformed_test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE).map(vectorize_text)

predictions = model.predict(transformed_test_ds)
score = tf.nn.sigmoid(predictions)

iterator = iter(test_ds)
for i in range(15):
    text, _ = iterator.get_next()
    print("{}, {}".format(score[i], text))
