
"""Experiment 17: Simple CNN for binary classification (cats vs dogs style).
Input: folder 'data/train' with subfolders 'class0' and 'class1' or use keras preprocessing to load images.
Output: model summary and final train/val accuracy printed and saved history plot.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
if not os.path.exists("data/train"):
    print("Please prepare 'data/train' and 'data/val' directories with class subfolders before running this script.")
else:
    train_dir = "data/train"
    val_dir = "data/val"
    img_size = (64,64)
    batch=32

    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_dir, target_size=img_size, batch_size=batch, class_mode='binary')
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir, target_size=img_size, batch_size=batch, class_mode='binary')

    model = models.Sequential([
        layers.Conv2D(32,3,activation='relu', input_shape=img_size+(3,)),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(2, activation='softmax')  # softmax for 2 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_gen, validation_data=val_gen, epochs=5)
    acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final train acc: {acc:.4f}, val acc: {val_acc:.4f}")
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.savefig("exp17_cnn_loss.png")
    print("Saved loss plot to exp17_cnn_loss.png")
