
"""Experiment 19: UNet-like model for segmentation (toy).
Input: image folder with images and masks. Provide 'images/' and 'masks/' equal-sized pairs.
Output: predicted mask saved as 'exp19_pred_mask.png' and Dice score printed.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
import glob
import os

def conv_block(x, filters):
    x = layers.Conv2D(filters,3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters,3, padding='same', activation='relu')(x)
    return x

inp = layers.Input((128,128,3))
c1 = conv_block(inp,16)
p1 = layers.MaxPooling2D()(c1)
c2 = conv_block(p1,32)
u1 = layers.UpSampling2D()(c2)
concat = layers.Concatenate()([u1, c1])
out = layers.Conv2D(1,1,activation='sigmoid')(concat)
model = Model(inp, out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[])

if not os.path.exists("images/1.png") or not os.path.exists("masks/1.png"):
    print("Please place a sample pair images/1.png and masks/1.png before running.")
else:
    img = cv2.imread("images/1.png")
    mask = cv2.imread("masks/1.png", cv2.IMREAD_GRAYSCALE)
    img_res = cv2.resize(img,(128,128))/255.0
    mask_res = cv2.resize(mask,(128,128))/255.0
    mask_res = (mask_res>127).astype(np.float32)
    model.fit(np.expand_dims(img_res,0), np.expand_dims(mask_res,0), epochs=5, verbose=1)
    pred = model.predict(np.expand_dims(img_res,0))[0,:,:,0]
    pred_bin = (pred>0.5).astype(np.uint8)*255
    cv2.imwrite("exp19_pred_mask.png", pred_bin)
    intersection = np.sum((pred>0.5) & (mask_res>0.5))
    dice = 2*intersection / (np.sum(pred>0.5)+np.sum(mask_res>0.5)+1e-8)
    print("Dice score (sample):", dice)
    print("Saved predicted mask to exp19_pred_mask.png")
