import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
from skimage import io, color
import random
import numpy as np
import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import os
import time
import scipy
from scipy import ndimage
from skimage.segmentation import mark_boundaries
#from train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='SLIC')
    parser.add_argument('--epochs', type=int, default=10 , help="iterations")
    parser.add_argument('--clusters', type=int, default=30 , help="K clusters")
    parser.add_argument('--filename', type=str, default = "brandeis.jfif", help="file name")
    parser.add_argument('--m', type=int, default = 10, help="compactness")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    clusters = args.clusters
    K = np.indices((clusters,1))
    
    #partitions the clusters with the SLIC algorithm
    partitions = partition(args.epochs, clusters, args.filename, args.m)
    size = np.shape(partitions)
    A = np.zeros(size)
    F = io.imread(args.filename)
    #creates an image of just the partitions alone and the partitions on top of the original image
    for i in range(size[0]):
        for j in range(size[1]-1):
            k = partitions[i,j]
            if (partitions[i,j+1] != k):
                A[i,j] = 255
                A[i,j+1] = 255
                F[i,j,0] = 0
                F[i,j+1,0] = 0
                F[i,j,1] = 0
                F[i,j+1,1] = 0
                F[i,j,2] = 0
                F[i,j+1,2] = 0
    for j in range(size[1]):
        for i in range(size[0]-1):
            k = partitions[i,j]
            if (partitions[i+1,j] != k):
                A[i,j] = 255
                A[i+1,j] = 255 
                F[i,j,0] = 0
                F[i+1,j,0] = 0
                F[i,j,1] = 0
                F[i+1,j,1] = 0
                F[i,j,2] = 0
                F[i+1,j,2] = 0
    plt.imshow(A)
    F[:,:,0] = F[:,:,0] - A
    F[:,:,1] = F[:,:,1] - A
    F[:,:,2] = F[:,:,2] - A
    plt.figure()
    plt.imshow(F)
    plt.figure()
    img = io.imread(args.filename) 
    img = mark_boundaries(image=img, label_img=partitions.astype(int))
    plt.imshow(img)
    plt.figure() 
    plt.imshow(io.imread(args.filename))
    return

def partition(epochs: int, clusters: int, filename: str, m: int):

    I = io.imread("brandeis.jfif")
    size = np.shape(I[:,:,0])
    I = color.rgb2lab(I)
    print("Start training...")
    #initialize clusters
    K = np.zeros((clusters,5))
    for i in range(clusters):
        a = random.randint(0,size[0])
        b = random.randint(0,size[1])
        K[i,0] = a
        K[i,1] = b
        K[i,2] = I[a,b,0]
        K[i,3] = I[a,b,1]
        K[i,4] = I[a,b,2]
    partitions = np.zeros(size)
    S = math.sqrt((size[0]*size[1])/clusters)
    #creates the constant 
    p = m/S
    for n in range(epochs):
        tik = time.time()
        #for each pixel
        for i in range(size[0]):
            for j in range(size[1]):
                #calculates the closest cluster center and adds the pixel to that cluster
                closest = 0
                closestDistance = p*math.sqrt((K[0,0] - i)**2 + (K[0,1] - j)**2) + math.sqrt((K[0,2] - I[i,j,0])**2 + (K[0,3] - I[i,j,1])**2 + (K[0,4] - I[i,j,2])**2)
                for k in range(1,clusters):
                    newDistance = p*math.sqrt((K[k,0] - i)**2 + (K[k,1] - j)**2) + math.sqrt((K[k,2] - I[i,j,0])**2 + (K[k,3] - I[i,j,1])**2 + (K[k,4] - I[i,j,2])**2)
                    if (newDistance < closestDistance):
                        closestDistance = newDistance
                        closest = k
                partitions[i,j] = closest
        #calculates the new centers as the means of the old cluster
        for k in range(clusters):
            total = np.zeros(5)
            count = 0;
            for i in range(size[0]):
                for j in range(size[1]):
                    if (partitions[i,j] == k):
                        count = count + 1
                        total[0] = total[0] + i
                        total[1] = total[1] + j
                        total[2] = total[2] + I[i,j,0]
                        total[3] = total[3] + I[i,j,1]
                        total[4] = total[4] + I[i,j,2]
            if (count == 0):
                print("Unused cluster: %d"%(k))
                
            else:
                K[k, 0] = round(total[0]/count)
                K[k, 1] = round(total[1]/count)
                K[k, 2] = round(total[2]/count)
                K[k, 3] = round(total[3]/count)
                K[k, 4] = round(total[4]/count)
        elapse = time.time() - tik
        print("Epoch: [%d/%d]; Time: %.2f" % (n + 1, epochs, elapse))
    return partitions
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    #torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))
if __name__ == "__main__":
    main()
