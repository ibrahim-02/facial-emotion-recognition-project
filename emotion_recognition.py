import sys
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv('fer2013.csv')

# Drop rows with NaN values in the 'Usage' column
df = df.dropna(subset=['Usage'])

# Data preprocessing
X_train, train_y, X_test, test_y = [], [], [], []
max_length = 48 * 48  # Set the maximum length of pixel values

for index, row in df.iterrows():
    val = [int(x) for x in row['pixels'].split(" ") if x]  # Filter out empty strings
    val = pad_sequences([val], maxlen=max_length, dtype='float32')[0]  # Pad or truncate the pixel values
    print(row['Usage'])
    try:
        if 'Training' in row['Usage']:
            X_train.append(val)
            train_y.append(row['emotion'])

    except:
        print(f"Error occurred at index: {index} and row: {row}")

# Convert labels to numeric values
train_y = np.array(train_y, dtype='float32')


# One-hot encode labels
num_labels = 7
train_y = to_categorical(train_y, num_classes=num_labels, dtype='uint8')


# Convert data to numpy arrays
X_train = np.array(X_train, dtype='float32')


# Normalize data
X_train /= 255.0


# Reshape data
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)


# Build the CNN model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 30

if len(X_train) > 0 and len(train_y) > 0:
    model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, shuffle=True)
else:
    print("Input data is empty.")

# Save the model
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")