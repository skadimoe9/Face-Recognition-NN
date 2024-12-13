import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def flatten_image(input_directory):
    # List all folders in the input directory
    folders = [f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))]
    print(f"Found folders: {folders}")

    # Initialize arrays for inputs and outputs
    X = []
    y = []

    # Create a mapping from folder names to one-hot encoded labels
    label_map = {folder: idx for idx, folder in enumerate(folders)}
    num_classes = len(folders)

    # Process each folder and photo
    for folder in folders:
        folder_path = os.path.join(input_directory, folder)
        
        # Sort the list of photos to ensure consistent order
        photos = sorted([p for p in os.listdir(folder_path) if p.endswith(('.png', '.jpg', '.jpeg'))])
        
        for photo in photos:
            photo_path = os.path.join(folder_path, photo)
            image = Image.open(photo_path)
            
            # Convert the image to a numpy array and flatten it
            image_array = np.array(image).flatten()
            X.append(image_array)
            
            # Create a one-hot encoded label
            label = np.zeros(num_classes)
            label[label_map[folder]] = 1
            y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

class NormalizeImage:
    def __init__(self, data):
        self.data = data

    def transform(self):
        #modified so the range of normalized data is [-1, 1]
        return self.data/255

    def inverse_transform(self):
        return self.data*255

def split_train_test(combined_array):
    np.random.shuffle(combined_array)
    # Calculate the split indices
    num_samples = combined_array.shape[0]
    train_end = int(0.7 * num_samples) # 70% of the data is used for training
    test_end = int(0.85 * num_samples) # 15% of the data is used for testing

    # Split the data into training, testing, and validation sets
    train_data = combined_array[:train_end]
    test_data = combined_array[train_end:test_end]
    val_data = combined_array[test_end:]

    return train_data, test_data, val_data

def split_input_output(data, num_input_features):
    X_data = data[:, :num_input_features]
    y_data = data[:, num_input_features:]
    return X_data, y_data

def process_all(input_directory):
    X, y = flatten_image(input_directory)

    scalerinput = NormalizeImage(X)
    X_normalized = scalerinput.transform()
    combined_array = np.hstack((X_normalized, y))

    train_data, test_data, val_data = split_train_test(combined_array)
    num_input_features = X.shape[1]
    X_train, y_train = split_input_output(train_data, num_input_features)
    X_test, y_test = split_input_output(test_data, num_input_features)
    X_val, y_val = split_input_output(val_data, num_input_features)

    return X_train, y_train, X_test, y_test, X_val, y_val, scalerinput