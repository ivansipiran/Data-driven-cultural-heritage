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

shapenet_part_dir = '../data/shapenetcore_part'

class ShapeNetDataset(data.Dataset):
    def __init__(self, root_dir = shapenet_part_dir, npoints = 2048, do_holes = True, function = None, class_choice = None, split = 'train', hole_size=0.35):
        self.npoints = npoints
        self.root = root_dir
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = False
        self.normalize = normalize
        self.do_holes = do_holes
        self.npoints = npoints
        self.hole_size = hole_size

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
            print(self.cat)
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                sys.exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),self.cat[item], token))            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
    
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]

        point_set = np.loadtxt(fn[1]).astype(np.float32)
        point_set = resample_pcd(point_set, self.npoints)
        point_set = rotate_pcd_shapeNet(point_set)

        if self.normalize:
            point_set = normalize(point_set, unit_ball = True)
        foldername = fn[3]
        filename = fn[4]
        
        if self.do_holes:
            partial, hole = make_holes_pcd_2(point_set, hole_size=self.hole_size)
        else:
            partial = point_set
            hole    = point_set
        return filename, resample_pcd(partial, self.npoints), resample_pcd(hole, self.npoints // 2), resample_pcd(point_set, self.npoints)
            
    def __len__(self):
        return len(self.datapath)
