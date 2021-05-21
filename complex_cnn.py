import tensorflow as tf
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# raw_train = tf.data.TFRecordDataset("/content/drive/My Drive/ML/proj1/nsynth-train.tfrecord")
raw_valid = tf.data.TFRecordDataset("/content/drive/My Drive/ML/proj1/nsynth-valid.tfrecord")
raw_test = tf.data.TFRecordDataset("/content/drive/My Drive/ML/proj1/nsynth-test (1).tfrecord")

def _parseme(raw_audio_record):
    feature_description = {
        'note': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'note_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'instrument': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'instrument_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'pitch': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'velocity': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'sample_rate': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'audio': tf.io.FixedLenSequenceFeature([], tf.float32,  allow_missing=True, default_value=0.0),
        'qualities': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'qualities_str': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=''),
        'instrument_family': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'instrument_family_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'instrument_source': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'instrument_source_str': tf.io.FixedLenFeature([], tf.string, default_value='')
    }

    return tf.io.parse_single_example(raw_audio_record, feature_description)


# train = raw_train.map(_parseme)
valid = raw_valid.map(_parseme)
test = raw_test.map(_parseme)

x_train = []
y_train = []
x_valid = []
y_valid = []
x_test = []
y_test = []

# for entry in train:
#     n = int(entry["instrument_family"])
#     if n == 9:
#         continue
#     if n > 9:
#       n = n-1
#     temp = [0]*10
#     temp[n] = 1
#     x_temp = scipy.signal.resample(entry["audio"], 8000)
#     x_train.append(x_temp)
#     y_train.append(temp)

# x_train = np.asarray(x_train)
# x_train_new = x_train[:, :, np.newaxis]

for entry in valid:
    n = int(entry["instrument_family"])
    if n == 9:
        continue
    if n > 9:
      n = n-1
    temp = [0]*10
    temp[n] = 1
    x_temp = scipy.signal.resample(entry["audio"], 8000)
    x_valid.append(x_temp)
    y_valid.append(temp)

x_valid = np.asarray(x_valid)
x_valid_new = x_valid[:, :, np.newaxis]


for entry in test:
    n = int(entry["instrument_family"])
    if n == 9:
        continue
    if n > 9:
      n = n-1
    temp = [0] * 10
    temp[n] = 1
    x_temp = scipy.signal.resample(entry["audio"], 8000)
    x_test.append(x_temp)
    y_test.append(temp)

x_test = np.asarray(x_test)
x_test_new = x_test[:, :, np.newaxis]


# final complex model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D( filters=64, kernel_size=2, input_shape=(8000, 1),activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.Conv1D( filters=32, kernel_size=2,activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.Conv1D(filters=92, kernel_size=2,activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(24, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])


# history = model.fit(x_train_new, np.asarray(y_train),validation_data=(x_valid_new,np.asarray(y_valid)) , epochs=40)

# delete the next line
history = model.fit(x_valid_new, np.asarray(y_valid) , epochs=40)


model.save('cnn_complex.h5')

score = model.evaluate(x_test_new, np.asarray(y_test))
print("accuracy = " + str(score[1]))