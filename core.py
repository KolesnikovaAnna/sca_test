import os
import time
from datetime import timedelta

import optuna
import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .dataloader import CustomDataGenerator

def run_sca(data_path, model_fn, augmentations_list=None, n_trials=20, batch_size=8, epochs=5, img_size=(224, 224)):
    datagen_val = ImageDataGenerator(rescale=1. / 255)
    val_generator = datagen_val.flow_from_directory(
        directory=os.path.join(data_path, 'train/val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    class_indices = val_generator.class_indices

    def objective(trial):
        brightness = trial.suggest_float("brightness_limit", 0.0, 0.5)
        contrast = trial.suggest_float("contrast_limit", 0.0, 0.5)
        rotate = trial.suggest_int("rotate_limit", 0, 45)
        hue = trial.suggest_int("hue_shift_limit", 0, 30)
        sat = trial.suggest_int("sat_shift_limit", 0, 50)
        val_shift = trial.suggest_int("val_shift_limit", 0, 50)

        if augmentations_list is None:
            augmentations = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=0.5),
                A.Rotate(limit=rotate, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val_shift, p=0.5),
                A.Resize(*img_size)
            ])
        else:
            augmentations = A.Compose([
                aug(p=0.5) for aug in augmentations_list
            ] + [A.Resize(*img_size)])

        train_generator = CustomDataGenerator(
            directory=os.path.join(data_path, 'train/train'),
            batch_size=batch_size,
            augmentations=augmentations,
            class_indices=class_indices,
            img_size=img_size
        )

        base_model = model_fn()
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(len(class_indices), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_generator,
                  validation_data=val_generator,
                  epochs=epochs,
                  verbose=0,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])

        _, val_acc = model.evaluate(val_generator, verbose=0)
        return val_acc

    print("🔍 Starting Smart Composite Augmentation...")
    start = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    duration = timedelta(seconds=int(time.time() - start))

    print("\n✅ Optimization finished.")
    print("Best trial:")
    print(study.best_trial)
    print(f"⏱️ Total optimization time: {duration}")
