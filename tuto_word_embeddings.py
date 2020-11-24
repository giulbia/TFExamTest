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


# PREP DATA #
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 1024

max_features = 5000  # Maximum vocab size.
max_len = 128  # Sequence length to pad the outputs to
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
test_ds = test_ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE).map(vectorize_text)

# MODEL #

model = tf.keras.Sequential([
    keras.layers.Embedding(max_features + 1, embedding_dim),
    keras.layers.Dropout(0.2),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)])

model.summary()

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
