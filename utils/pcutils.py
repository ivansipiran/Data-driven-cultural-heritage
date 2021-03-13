import numpy as np
import random   
from numpy.linalg import norm
import open3d as o3d
import trimesh
from numpy.random import randint
from numpy import linalg as LA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import math

from matplotlib.colors import LightSource


def origin_mass_center(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd

def normalize(points, unit_ball = False):
    normalized_points = origin_mass_center(points)
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)
    if unit_ball:
        normalized_points = normalized_points/(max_distance)
    else:
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points

def normalize2(points, unit_ball = False):
    normalized_points = origin_mass_center(points)
    normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points, scale
    #return normalized_points

def augmented_normalize(points, unit_ball = False, rand_shift = 0):
    normalized_points = origin_mass_center(points)
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)
    if unit_ball:
        normalized_points = normalized_points/(max_distance)
    else:
        normalized_points = normalized_points/((2 + rand_shift) * max_distance)

    return normalized_points

def show_pcd(X_iso, show=True):
    lim = 0.8
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    sequence_containing_x_vals = list(X_iso[:, 0:1]) 
    sequence_containing_y_vals = list(X_iso[:, 1:2]) 
    sequence_containing_z_vals = list(X_iso[:, 2:3])

    C = np.array(['#ff0000' for i in range(len(X_iso))])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=C, s=10, depthshade=True) 
    ax.grid(False)
    ax.axis(False)
    if show:
        pyplot.show()
    

def show_pcd2(X_iso):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = list(X_iso[:, 0:1]) 
    sequence_containing_y_vals = list(X_iso[:, 1:2]) 
    sequence_containing_z_vals = list(X_iso[:, 2:3])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals) 
    pyplot.show()
    
    
def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]


def make_holes_pcd(pcd, hole_size=0.1):
    """[summary]
    
    Arguments:
        pcd {[float[n,3]]} -- [point cloud data of n size in x, y, z format]
    
    Returns:
        [float[m,3]] -- [point cloud data in x, y, z of m size format (m < n)]
    """
    rand_point = pcd[randint(0, pcd.shape[0])]

    partial_pcd = []
    
    for i in range(pcd.shape[0]):
        dist = np.linalg.norm(rand_point - pcd[i])  
        if dist >= hole_size:
            # pcd.vertices[i] = rand_point
            partial_pcd = partial_pcd + [pcd[i]]
    return np.array([np.array(e) for e in partial_pcd])   

def make_holes_pcd_2(pcd, hole_size=0.1):

    rand_point = pcd[randint(0, pcd.shape[0])]
    #print(rand_point)

    partial_pcd = []
    hole_pcd = []
    
    for i in range(pcd.shape[0]):
        dist = np.linalg.norm(rand_point - pcd[i])
        if dist >= hole_size:
            # pcd.vertices[i] = rand_point
            partial_pcd = partial_pcd + [pcd[i]]
        else:
            hole_pcd = hole_pcd + [pcd[i]]

    partial_pcd = np.array([np.array(e) for e in partial_pcd])
    hole_pcd = np.array([np.array(e) for e in hole_pcd])
    # print(f'Holes: input_shape {pcd.shape} - partial_shape {partial_pcd.shape} - hole_shape {hole_pcd.shape}')
    return partial_pcd, hole_pcd



def make_holes_base(pcd, range_p):
    min_h = 10000
    max_h = -10000
    min_pos = 0
    max_pos = 0
    for i in range(pcd.shape[0]):
        if pcd[i][2] > max_h:
            max_h = pcd[i][2]
            max_pos = i
        if pcd[i][2] < min_h:
            min_h = pcd[i][2]
            min_pos = i
            
    range_h = abs(max_h - min_h)
    
    partial_pcd = []
    hole_pcd = []
    
    max_h = random.uniform(range_p[0], range_p[1])
    for i in range(pcd.shape[0]):
        if pcd[i][2] >= (min_h + range_h * max_h):
            # pcd.vertices[i] = rand_point
            partial_pcd = partial_pcd + [pcd[i]]
        else:
            hole_pcd = hole_pcd + [pcd[i]]
    return np.array([np.array(e) for e in partial_pcd]), np.array([np.array(e) for e in hole_pcd])
    



def make_holes_horizontally(pcd, range_p):
    min_h = 10000
    max_h = -10000
    min_pos = 0
    max_pos = 0
    for i in range(pcd.shape[0]):
        if pcd[i][2] > max_h:
            max_h = pcd[i][2]
            max_pos = i
        if pcd[i][2] < min_h:
            min_h = pcd[i][2]
            min_pos = i
            
    range_h = abs(max_h - min_h)

    height_begin = random.uniform(min_h, max_h - (range_h / 7)) 
    height_end = height_begin + random.uniform(range_h / 7, range_h / 4)
    
    partial_pcd = []
    hole_pcd = []
    
    for i in range(pcd.shape[0]):
        if((pcd[i][2] > height_begin) and (pcd[i][2] < height_end)):
            hole_pcd = hole_pcd + [pcd[i]]
        else:
            partial_pcd = partial_pcd + [pcd[i]]
    return np.array([np.array(e) for e in partial_pcd]), np.array([np.array(e) for e in hole_pcd])
    


def make_holes_pcd_3(pcd, range_hole):
    two_holes =  bool(random.getrandbits(1))
    
    rand_point1 = pcd[randint(0, pcd.shape[0])]
    hole_size1 = random.uniform(range_hole[0], range_hole[1])
    
    rand_point2 = pcd[randint(0, pcd.shape[0])]
    hole_size2 = random.uniform(range_hole[0], range_hole[1])
    
    
    
    
    partial_pcd = []
    
    hole_pcd = []
    
    for i in range(pcd.shape[0]):
        if not two_holes:
            dist = np.linalg.norm(rand_point1 - pcd[i])
            if dist >= hole_size1:
                # pcd.vertices[i] = rand_point
                partial_pcd = partial_pcd + [pcd[i]]
            else:
                hole_pcd = hole_pcd + [pcd[i]]
            
        else:
            dist1 = np.linalg.norm(rand_point1 - pcd[i])
            dist2 = np.linalg.norm(rand_point2 - pcd[i])
            if ((dist1 < hole_size1) or  (dist2 < hole_size2)):
                hole_pcd = hole_pcd + [pcd[i]]
            else:
                partial_pcd = partial_pcd + [pcd[i]]
    return np.array([np.array(e) for e in partial_pcd]), np.array([np.array(e) for e in hole_pcd])




def read_pcd(filename, n_points = 0):
    
    if filename[filename.rfind('.') + 1:] == 'pcd':
        pcd = o3d.io.read_point_cloud(filename)
        pcd = np.array(pcd.points)
        if n_points != 0:
            pcd = resample_pcd(pcd, n_points)
    else:
        pcd = trimesh.load(filename).sample(n_points)
    return pcd




def get_rotation_x(teta):
    return np.array([
        np.array([1,    0,                0]),
        np.array([0,    math.cos(teta),   -math.sin(teta)]),
        np.array([0,    math.sin(teta),    math.cos(teta)])
    ])

def get_rotation_y(teta):
    return np.array([
        np.array([math.cos(teta),  0,       math.sin(teta)]),
        np.array([0,               1,       0]),
        np.array([-math.sin(teta), 0,       math.cos(teta)])
    ])


def get_rotation_z(teta):
    return np.array([
        np.array([math.cos(teta),  -math.sin(teta),       0]),
        np.array([math.sin(teta),  math.cos(teta),        0]),
        np.array([0,               0,                     1])
    ])


def add_rotation_to_pcloud(pcloud, r_rotation):
    # r_rotation = rand_rotation_matrix()
    if len(pcloud.shape) == 2:
        return pcloud.dot(r_rotation)
    else:
        return np.asarray([e.dot(r_rotation) for e in pcloud])

def mean_min_square_distance(source, target):
    dists = []
    
    tree = o3d.geometry.KDTreeFlann(target.T)

    for i in range(source.shape[0]):
        [k, idx, d] = tree.search_knn_vector_3d(source[i].T, 1)
        dists.append(d)

    return np.mean(np.asarray(dists))


def save_point_cloud(filename, pcd):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(filename, pc)
