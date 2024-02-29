# Juan Irving Vasquez-Gomez
# jivg.org

from typing import Any
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def showGrid(grid, direction, nbv = None, predicted_nbv = None):
    # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (31,31,31))
    #grid3d = grid
    unknown = grid3d== 0.5 #np.logical_and((grid3d > 0.4),(grid3d <= 0.6))
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'

    # and plot everything
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
     
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    if nbv is not None:
        position = nbv[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=5.0, normalize=True, color = 'g')  
    
    if predicted_nbv is not None:
        position = predicted_nbv[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=5.0, normalize=True, color = 'r')
    ###QUITAR (comentar/descomentar) solo si necesitas quitar el grid
    # Hide grid lines
    #ax.grid(visible=None)

    # Hide axes ticks
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    #plt.axis('off')
    ###
    plt.savefig(direction, bbox_inches = 'tight')
    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.show()
    
    
    
def showGrid4(grid, nbv = None, predicted_nbv = None):
     # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (32,32,32))

    unknown = (grid3d == 0.5)
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'
    
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    if nbv is not None:
        position = nbv[:3]
        position_nbv = position * (scale / rate_voxel_map_sphere) + center
        direction_nbv = center - position_nbv
    if predicted_nbv is not None:
        position = predicted_nbv[:3]
        position_pred = position * (scale / rate_voxel_map_sphere) + center
        direction_pred = center - position_pred    
    
    fig = plt.figure(figsize=(12, 12))
    
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')

        
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')
    ax.view_init(elev=0.0, azim=0.0)

 
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')
    ax.view_init(elev=0.0, azim=90.0)
    
 
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=5.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=5.0, normalize=True, color = 'r')
    ax.view_init(elev=90.0, azim=0.0)
    

    
class NBVClassificationDatasetFull(Dataset):
    """NBV dataset."""
    def __init__(self, grid_file, nbv_class_file, transform=None):
        """
        Args:
            poses_file (string): Path to the sensor poses (next-best-views).
            root_dir (string): Directory with all the grids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.root_dir = root_dir
        self.grid_data = np.load(grid_file)
        self.nbv_class_data = np.load(nbv_class_file)
        self.transform = transform

    def __len__(self):
        return len(self.nbv_class_data)

    def __getitem__(self, idx):
        grid = self.grid_data[idx] 
        nbv_class = self.nbv_class_data[idx]
        sample = {'grid': grid, 'nbv_class': nbv_class}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class To3DGrid(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv_class = sample['grid'], sample['nbv_class']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        grid = np.reshape(grid, (31,31,31))
        
        return {'grid': grid,
                'nbv_class': nbv_class}

class ToSO3(object):
    """Convert angles to SO3 matrix
        Its organized as x,y,z,r11,...,r33"""
    def __call__(self, sample):

        grid, nbv_class = sample['grid'], sample['nbv_class']
        x = nbv_class[0]
        y = nbv_class[1]
        z = nbv_class[2]

        nbv_class = np.asarray([x,y,z], dtype='float64')
        return {'grid': grid,
                'nbv_class': nbv_class}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv_class = sample['grid'], sample['nbv_class']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        return {'grid': torch.from_numpy(np.array([grid])),
                'nbv_class': torch.from_numpy(np.array(nbv_class))}
                #'nbv_class': torch.tensor(nbv_class[0], dtype=torch.int64)}
