import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np

data, metadata = tfds.load('mnist', as_supervised=True, with_info=True)

training_data, test_data = data['train'], data['test']
class_names = metadata.features['label'].names

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

training_data = training_data.map(normalize)
test_data = test_data.map(normalize)

training_data = training_data.cache()
test_data = test_data.cache()

# Display the first 25 images from the training set
plt.figure(figsize=(10, 10))
plt.title('First 25 images from the training set')
for i, (image, label) in enumerate(training_data.take(25)):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    # Softmax is used for multi-class classification and returns a vector of values that represent the probability of each class
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

amount_training_data = metadata.splits['train'].num_examples
amount_test_data = metadata.splits['test'].num_examples

# Train with random data and repeat the data
BATCH_SIZE = 50
training_data = training_data.repeat().shuffle(amount_training_data).batch(BATCH_SIZE)
test_data = test_data.batch(amount_test_data)

history = model.fit(training_data, epochs = 6, steps_per_epoch = math.ceil(amount_training_data / BATCH_SIZE))
test_loss, test_accuracy = model.evaluate(test_data)

# Print the accuracy and loss of the model
figure, axis = plt.subplots(1, 2)

axis[0].plot(history.history['accuracy'])
axis[0].set_title('Model accuracy')
axis[0].set_ylabel('Accuracy')
axis[0].set_xlabel('Epoch')
axis[0].legend(['Train', 'Test'], loc='upper left')

axis[1].plot(history.history['loss'])
axis[1].set_title('Model loss')
axis[1].set_ylabel('Loss')
axis[1].set_xlabel('Epoch')
axis[1].legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.close()

# Predict the first 16 images from the test set
for test_images, test_labels in test_data.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

# Display the first 16 images from the test set and their predicted labels
plt.figure(figsize=(12, 12))
plt.title('First 16 images from the test set and their predicted labels')
plt.axis('off')
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = test_images[i].reshape((28, 28))
    plt.imshow(image, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    color = 'green' if predicted_label == test_labels[i] else 'red'
    plt.xlabel(f"Pred: {predicted_label}\nReal: {test_labels[i]}", color=color)

plt.tight_layout()
plt.show()
plt.close()
