import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class CustomDataGenerator(Sequence):
    def __init__(self, directory, batch_size, augmentations, class_indices, img_size=(224, 224)):
        self.image_paths = []
        self.labels = []
        self.batch_size = batch_size
        self.augment = augmentations
        self.class_indices = class_indices
        self.img_size = img_size

        for class_name in sorted(os.listdir(directory)):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir): continue
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_indices[class_name])

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_paths = [self.image_paths[i] for i in self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_labels = [self.labels[i] for i in self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]]

        images = []
        for img_path in batch_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                img = self.augment(image=img)["image"]
            except:
                continue

            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)

        images = np.stack(images)
        labels = tf.keras.utils.to_categorical(batch_labels[:len(images)], num_classes=len(self.class_indices))
        
        return images, labels
