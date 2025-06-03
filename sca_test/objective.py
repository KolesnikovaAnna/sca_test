import albumentations as A
import cv2
from sca_test.dataloader import CustomDataGenerator

def create_objective(
    dataset_path,
    train_dir,
    val_dir,
    batch_size,
    epochs,
    class_indices,
    use_random_brightness_contrast,
    use_hue_saturation_value,
    use_rotate,
    model,
    val_generator
):
    def objective(trial):
        print("\n[START] Trial", trial.number)
        transforms = []

        if use_random_brightness_contrast:
            brightness = trial.suggest_float("brightness_limit", 0.0, 0.5)
            contrast = trial.suggest_float("contrast_limit", 0.0, 0.5)
            transforms.append(
                A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=0.5)
            )

        if use_rotate:
            rotate = trial.suggest_int("rotate_limit", 0, 45)
            transforms.append(
                A.Rotate(limit=rotate, border_mode=cv2.BORDER_REFLECT_101, p=0.5)
            )

        if use_hue_saturation_value:
            hue = trial.suggest_int("hue_shift_limit", 0, 30)
            sat = trial.suggest_int("sat_shift_limit", 0, 50)
            val = trial.suggest_int("val_shift_limit", 0, 50)
            transforms.append(
                A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val, p=0.5)
            )

        transforms.append(A.Resize(224, 224))
        augmentations = A.Compose(transforms)

        print("[INFO] Creating train generator...")

        train_generator = CustomDataGenerator(
            directory=f"{dataset_path}/{train_dir}",
            batch_size=batch_size,
            augmentations=augmentations,
            class_indices=class_indices
        )

        print("[INFO] Train generator created.")
        
        print("[INFO] Compiling model...")
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("[INFO] Model compiled.")

        print("[INFO] Starting training...")
        x_val, y_val = val_generator[0]
        print(f"[DEBUG] val_generator[0] shapes: {x_val.shape}, {y_val.shape}")
        model.fit(
            train_generator,
            #validation_data=val_generator,
            epochs=epochs,
            verbose=1,
        )
        print("[INFO] Training complete.")

        val_loss, val_acc = model.evaluate(val_generator, verbose=0)
        return val_acc

    return objective
