import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import load_img, img_to_array

# Step 1: Load the Dataset
# Define the paths to the dataset
base_dir = 'path/to/dataset'  # Update this to your dataset path
cat_dir = os.path.join(base_dir, 'cats')
dog_dir = os.path.join(base_dir, 'dogs')

# Initialize lists to hold the images and labels
images = []
labels = []

# Load cat images
for filename in os.listdir(cat_dir):
    img_path = os.path.join(cat_dir, filename)
    img = load_img(img_path, target_size=(128, 128))  # Resize to 128x128
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    images.append(img_array)
    labels.append(0)  # 0 for cats

# Load dog images
for filename in os.listdir(dog_dir):
    img_path = os.path.join(dog_dir, filename)
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    images.append(img_array)
    labels.append(1)  # 1 for dogs

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# Step 2: Flatten the images for SVM
X_flat = X.reshape(X.shape[0], -1)

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Step 4: Train the SVM Classifier
model = svm.SVC(kernel='linear')  # You can choose different kernels like 'rbf'
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Optional: Display some test images with predictions
def display_images(images, labels, predictions):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {"Dog" if labels[i] else "Cat"}\nPred: {"Dog" if predictions[i] else "Cat"}')
        plt.axis('off')
    plt.show()

# Display some test images with predictions
display_images(X_test.reshape(-1, 128, 128, 3), y_test, y_pred)