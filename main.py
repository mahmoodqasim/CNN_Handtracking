import tensorflow as tf


from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

import time
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

def load_images_from_folder(folder):
    images = []
    names = []
    for i in range(1,5001):
        names.append(str(i) + '.png')
    for file in names:
        img = cv2.imread(os.path.join(folder,file),0)
        if img is not None:
            images.append(img)
    return images

# Load images from 'Data' folder
train_features = load_images_from_folder('Data')
train_features = np.array(train_features)

# Flatten from 5000, 200, 200 to 5000 * 40000
new = []
for i in range(len(train_features)):
    new.append(train_features[i].flatten())
train_features = np.array(new)

# Add in labeling for the 5 gestures
label = np.ones(len(train_features), dtype=int)
label[0:1000] = 0
label[1001:2000] = 1
label[2001:3000] = 2
label[3001:4000] = 3
label[4001:5000] = 4

# Shuffle them up
idx = np.random.permutation(len(train_features))
train_features, label = train_features[idx], label[idx]

# # Merge features with labels
# train_data = [train_features, label]
#
# # Getting the features and labels separated, kind of stupid cause I just added them together lol
# (X, y) = (train_data[0],train_data[1])

# Defining number of class, rows/cols
num_classes = len(np.unique((label)))
img_rows = 200
img_cols = 200


# Splitting into training / testing by 20%
X_train, X_test, y_train, y_test = train_test_split(train_features, label, test_size=0.2, random_state=64)

# Shape was (4000, 200, 200) and (1000, 200, 200) now, reshape to (4000, 1, 200, 200) and (1000, 1, 200, 200)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')

# Making it easier to compute, dividing by the max pixel value
X_train /= 255
X_test /= 255

# Convert class labels to binary class labels (ONE HOT ENCODING)
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# Defining the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(1, 200, 200)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train the model using 500 samples, 20% for validation, batch size 20
start = time.time()
model_info = model.fit(X_train[1:500], Y_train[1:500], batch_size=20, epochs=3, validation_split=0.2)
end = time.time()

# Evaluate the model using testing data
model.evaluate(X_test[1:200], Y_test[1:200])

# plot model history
plot_model_history(model_info)
print ("Model took this many seconds to train", (end - start))

# # compute test accuracy
# print ("Accuracy on test data is: ", accuracy(X_test, Y_test, model))

