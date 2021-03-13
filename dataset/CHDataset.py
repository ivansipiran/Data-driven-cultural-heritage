import open3d as o3d
import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
import trimesh
import sys
from os.path import join
from numpy import linalg as LA
import json
sys.path.append("./utils/")
from pcutils import normalize, make_holes_pcd_2, make_holes_pcd_3, make_holes_base, get_rotation_x, get_rotation_z, add_rotation_to_pcloud, make_holes_horizontally, augmented_normalize

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def rotate_pcd_shapeNet(pcd, posA = 1, posB = 2):
    n = pcd.shape[0]
    for i in range(n):
        temp = pcd[i][posA]
        pcd[i][posA] = pcd[i][posB]
        pcd[i][posB] = temp
    return pcd

class CHDataset(data.Dataset): 
    def __init__(self, complete_dir, holes_dir, complete_list, n_partial_models, npoints = 2048, do_holes = True, function = None):
        
        with open(complete_list) as file:
            self.relative_paths = [(line.strip()[:line.rfind('/')].strip('/').strip("."), line.strip()[line.rfind('/') + 1: ].strip('/')) for line in file]
        
        self.train_dir = holes_dir
        self.complete_dir = complete_dir

        self.npoints = npoints
        self.n_partial_models = n_partial_models

        random.shuffle(self.relative_paths)
        self.len = len(self.relative_paths) * n_partial_models
        
        self.do_holes = do_holes
        self.function = function

    def __getitem__(self, index):

        (rpath, name) = self.relative_paths[index // self.n_partial_models]
        partial_id = index % self.n_partial_models

        
        def read_pcd(filename, n_points = 0):
            
            if filename[filename.rfind('.') + 1:] == 'pcd':
                pcd = o3d.io.read_point_cloud(filename)
                pcd = np.array(pcd.points)
                if n_points != 0:
                    pcd = resample_pcd(pcd, n_points)
            else:
                pcd = trimesh.load(filename).sample(n_points)
            return pcd

        model_dir_name = name[:name.find('.')]

        # complete = trimesh.load(self.complete_dir + "/" + join( rpath, name)).sample(self.npoints )
        complete = read_pcd(self.complete_dir + "/" + join( rpath, name), self.npoints)


        
        if self.do_holes:
            scale_shift = random.uniform(-0.5, 0.5)
            complete = augmented_normalize(complete, unit_ball= False, rand_shift = scale_shift)

            # rotations 
            rot_z = get_rotation_z(np.deg2rad(random.uniform(0, 360)))
            rot_x = get_rotation_x(np.deg2rad(random.uniform(0, 15)))
            rotation_mat = np.dot(rot_x, rot_z)
            complete = add_rotation_to_pcloud(complete, rotation_mat)

            # holes
            partial, hole = make_holes_base(complete, [0.05, 0.175])
            #partial, hole = make_holes_pcd_3(complete, [0.05, 0.15])

            # translations
            # print(f'Partial shape: {partial.shape} - complete shape: {complete.shape}')
            z_shift = random.uniform(-0.3, 0.3)
            partial = partial + np.array([0, 0, z_shift])
            hole = hole + np.array([0, 0, z_shift])
            complete = complete + np.array([0, 0, z_shift])

            x_shift = random.uniform(-0.03, 0.03)
            partial = partial + np.array([x_shift, 0, 0])
            hole = hole + np.array([x_shift, 0, 0])
            complete = complete + np.array([x_shift, 0, 0])

            y_shift = random.uniform(-0.03, 0.03)
            partial = partial + np.array([0, y_shift, 0])
            hole = hole + np.array([0, y_shift, 0])
            complete = complete + np.array([0, y_shift, 0])

            
            #partial = add_rotation_to_pcloud(partial, rotation_mat)
            #hole = add_rotation_to_pcloud(hole, rotation_mat)
        else:
            complete = normalize(complete, unit_ball = False)
            partial = complete
            hole = complete
            

        #print(partial)
        #print(complete)
        #print(hole)
        return name, resample_pcd(partial, self.npoints), resample_pcd(hole, self.npoints //2 ), resample_pcd(complete, self.npoints )

    def __len__(self):
        return self.len

if __name__ == '__main__':

    dir_train = "./data/datasetCH/pottery_augmented_filtered"
    dir_test = "./data/datasetCH/pottery_augmented_filtered_test"
    holes_dir = ""

    complete_list_train = "./data/datasetCH/pottery_augmented_filtered_complete_train.txt"
    complete_list_test = "./data/datasetCH/pottery_augmented_filtered_complete_test.txt"

    dataset = CHDataset(dir_test, holes_dir, complete_list_test, 1, npoints=2048, do_holes=False)

    #min_z = 0
    #max_z = 0
    #for data in dataset:
    #    name, partial, hole, complete = data
        #print(f'partial({min(partial[:,2])},{max(partial[:,2])})  -> hole({min(hole[:,2])},{max(hole[:,2])}) -> complete({min(complete[:,2])},{max(complete[:,2])})')
    #    min_z = min_z + min(complete[:,2])
    #    max_z = max_z + max(complete[:,2])
    
    #print(min_z/len(dataset))
    #print(max_z/len(dataset))
    name, partial, hole, complete = dataset[0]
    save_point_cloud('complete.xyz', complete)
    save_point_cloud('partial.xyz', partial)
    save_point_cloud('hole.xyz', hole)
