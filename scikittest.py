# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:22:16 2024

@author: Administrator
"""
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io

img_rgb = io.imread("brandeis.jfif")
segments = slic(image=img_rgb, n_segments=30, convert2lab=True,enforce_connectivity=True)
plt.imshow(segments)
plt.figure()
img_rgb = mark_boundaries(image=img_rgb, label_img=segments)
plt.imshow(img_rgb)
plt.show()