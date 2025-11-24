
"""Experiment 18: Compare CNN training under varied batch size, optimizer, activation, lr.
Input: small image dataset with folders as in exp17
Output: accuracy per configuration printed
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(activation='relu'):
    m = models.Sequential([layers.Conv2D(16,3,activation=activation,input_shape=(64,64,3)),
                          layers.MaxPooling2D(),
                          layers.Flatten(),
                          layers.Dense(32, activation=activation),
                          layers.Dense(2, activation='softmax')])
    return m

configs = [
    {'batch':16, 'opt':'adam', 'act':'relu', 'lr':0.001},
    {'batch':32, 'opt':'sgd', 'act':'relu', 'lr':0.01},
]
train_dir = "data/train"; val_dir="data/val"
if not os.path.exists(train_dir):
    print("Please prepare 'data/train' and 'data/val' directories before running this script.")
else:
    for cfg in configs:
        train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_dir, target_size=(64,64), batch_size=cfg['batch'], class_mode='binary')
        val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir, target_size=(64,64), batch_size=cfg['batch'], class_mode='binary')
        m = build_model(cfg['act'])
        if cfg['opt']=='adam':
            opt = tf.keras.optimizers.Adam(cfg['lr'])
        else:
            opt = tf.keras.optimizers.SGD(cfg['lr'])
        m.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        h = m.fit(train_gen, validation_data=val_gen, epochs=3, verbose=0)
        print(f"Config {cfg} => train_acc {h.history['accuracy'][-1]:.4f}, val_acc {h.history['val_accuracy'][-1]:.4f}")
