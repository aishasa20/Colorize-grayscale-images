import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.layers import Conv2D, UpSampling2D, MaxPool2D
from keras.models import Sequential
from keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import resize


dataset = keras.datasets.cifar10.load_data()


(X_train, _), (X_test, _) = dataset


plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(X_train[i])
    plt.axis('off')
plt.show()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X = []
Y = []
for img in X_train:
    lab = rgb2lab(img)
    X.append(lab[:, :, 0])  # assign L channel to X
    Y.append(lab[:, :, 1:3] / 128)  # assign A and B channels to Y

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

# encoder
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))


# decoder
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=64, verbose=1)

for i in range(10):
    img1_color = []
    img = X_test[i]
    img1 = img_to_array(X_test[i])
    img1_color.append(img1)

    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(img1_color)[:, :, :, 0]
    img1_color = img1_color.reshape(1, 32, 32, 1)

    output1 = model.predict(img1_color)
    output1 = output1 * 128

    result = np.zeros((32, 32, 3))
    result[:, :, 0] = img1_color[0][:, :, 0]
    result[:, :, 1:] = output1[0]

    fig, axes = plt.subplots(1, 3 )

    axes[0].imshow(X_test[i])
    axes[0].title.set_text('Original Color Image')
    axes[0].axis('off')

    axes[1].imshow(result[:, :, 0], cmap='gray')
    axes[1].title.set_text('Grayscale image')
    axes[1].axis('off')

    axes[2].imshow(lab2rgb(result))
    axes[2].title.set_text('Colorized image')
    axes[2].axis('off')

    plt.show()
