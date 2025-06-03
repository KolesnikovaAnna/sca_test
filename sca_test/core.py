import os
import time
from datetime import timedelta

import optuna
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sca_test.dataloader import CustomDataGenerator
from sca_test.objective import create_objective

IMG_SIZE = (224, 224)

def run_sca(
    model=None,
    dataset_path='',
    n_trials=20,
    batch_size=8,
    epochs=5,
    use_random_brightness_contrast=False,
    use_hue_saturation_value=False,
    use_rotate=False,
    val_dir='train/val',
    train_dir='train/train'
):
    datagen_val = ImageDataGenerator(rescale=1./255)
    val_generator = datagen_val.flow_from_directory(
        directory=os.path.join(dataset_path, val_dir),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )
    class_indices = val_generator.class_indices
    print("\ndatagen_val finished")

    if model is None:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(class_indices), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

    print("\nmodel finished")

    objective = create_objective(
        dataset_path=dataset_path,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        epochs=epochs,
        class_indices=class_indices,
        use_random_brightness_contrast=use_random_brightness_contrast,
        use_hue_saturation_value=use_hue_saturation_value,
        use_rotate=use_rotate,
        model=model,
        val_generator=val_generator
    )
    print("\nobjective finished")

    class TrialCallback:
        def __init__(self):
            self.best_acc = 0.0
            self.best_params = {}

        def __call__(self, study, trial):
            if trial.value > self.best_acc:
                self.best_acc = trial.value
                self.best_params = trial.params.copy()

    callback = TrialCallback()

    print("🔍 Запуск оптимизации...")
    start_time = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    duration = timedelta(seconds=int(time.time() - start_time))
    print(f"\n✅ Оптимизация завершена за {duration}")
    print(f"🎯 Лучшая точность: {callback.best_acc:.4f}")
    print(f"🔧 Лучшие параметры: {callback.best_params}")

    return model, callback.best_params, callback.best_acc
