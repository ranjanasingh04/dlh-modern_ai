#!/usr/bin/env python3

from tensorflow import keras
import matplotlib.pyplot as plt
build_model = __import__('0-sequential').build_model


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

""""The code below visually displays the first 10 images in the training set with their corresponding labels."""
num_samples = 10
plt.figure(figsize=(10, 2))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

input_dim = x_train.shape[1]
n_h = 4
model = build_model(input_dim, n_h)

model.summary()

"""The code below prints the 28x28 grayscale pixel values of the first training image and displays it with a colorbar to show intensity."""
image = x_train[0]
label = y_train[0]

print("Pixel Values (28x28):")
for row in image:
    print(" ".join(f"{pixel:3}" for pixel in row))

plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.colorbar(label="Pixel Intensity")
plt.show()