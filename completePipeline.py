import argparse
import open3d as o3d
import numpy as np
import torch
from models.hole_residual import MSNmodel
from utils.utils import weights_init, visdom_show_pc, save_paths, save_model, vis_curve
from utils.pcutils import *
import sys
import os

#Function to read point cloud
def read_points(filename):
   
    geom = o3d.io.read_point_cloud(filename)
    points = np.asarray(geom.points)
    
    return points

#Smooth point cloud 
def guided_filter(pcd, flags, radius, epsilon):
    kdtree = o3d.geometry.KDTreeFlann(pcd.T)
    points_copy = np.array(pcd) 
    num_points = len(pcd)

    for i in range(num_points): 
        if flags[i] == 1: 
            continue
        
        k, idx, _ = kdtree.search_radius_vector_3d(pcd[i].T, radius)
        if k < 3: 
            continue
        
        neighbors = pcd[idx, :] 
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ pcd[i].T + b

    return points_copy

def guided_filter2(pcd, flags, radius, epsilon, number):
    kdtree = o3d.geometry.KDTreeFlann(pcd.T)
    points_copy = np.array(pcd) 
    num_points = len(pcd)

    new_points = []

    for i in range(num_points): 
        if flags[i] == 1: 
            continue
        
        k, idx, _ = kdtree.search_radius_vector_3d(pcd[i].T, radius)
        if k < 3: 
            continue
        
        neighbors = pcd[idx, :] 
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ pcd[i].T + b
        
    return points_copy, points_copy[number:,:]

def save_point_cloud(filename, pcd):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(filename, pc)

def consolidatePointCloud(pcdPartial, pcdHole, name):
    #Smooth the combination of point clouds
    flags = np.concatenate((np.ones((pcdPartial.shape[0],1), dtype='int32'), 2*np.ones((pcdHole.shape[0],1), dtype='int32')))
    points = np.concatenate((pcdPartial, pcdHole))
    points = guided_filter(points, flags, 0.1, 0.01)

    save_point_cloud(name, points)

#Remove point far from the bottom boundary in the input
def consolidatePointCloud2(pcdPartial, pcdHole):
    maxY = np.max(pcdPartial[:,1])
    minY = np.min(pcdPartial[:,1])

    threshold = minY + 0.1*(maxY - minY)

    new_points = []
    for i in range(pcdHole.shape[0]):
        if pcdHole[i,1] <= threshold:
            new_points.append(pcdHole[i])

    print(len(new_points))
    pointsHole = np.vstack(new_points)
    
    return pointsHole
    
#Perform the prediction of our neural network
def predict(model, pcd):
    #Transform the input point cloud to feed the neural network
    inputPartial = np.asarray(pcd.points).astype(np.float32)
    rot_x = get_rotation_x(np.deg2rad(-90))
    inputPartial = add_rotation_to_pcloud(inputPartial, rot_x)
    inputPartial = inputPartial.astype(np.float32)
    inputPartial, scal = normalize2(inputPartial, unit_ball=False)

    dummy_hole = np.zeros((1024,3)).astype(np.float32)
    dummy_complete = np.zeros((2048,3)).astype(np.float32)

    #Input data must be a tensor
    in_partial = torch.unsqueeze(torch.from_numpy(inputPartial), 0)
    in_complete= torch.unsqueeze(torch.from_numpy(dummy_complete), 0)
    in_hole= torch.unsqueeze(torch.from_numpy(dummy_hole), 0)

    #Sent to GPU
    in_partial = in_partial.to(device)
    in_complete = in_complete.to(device)
    in_hole = in_hole.to(device)

    #The inference happens in this block. The output of the model needs to go back to numpy arrays before returned
    with torch.no_grad():
        output, output2, rec_loss1, rec_loss2, exp_loss, spc, flags = model(in_partial, in_hole, in_complete, 0.005, 50)    
        flags2 = flags.cpu().numpy()[0]
        spc2 = spc.cpu().numpy()[0]
        pred = output2.cpu().numpy()[0]
        
        gt = in_complete.cpu().numpy()[0]
        partial = in_partial.cpu().numpy()[0]

        holeBefore = spc2[np.flatnonzero(flags2<0.1),:]
        holeAfter = pred[np.flatnonzero(flags2<0.1),:]

        partialBefore = spc2[np.flatnonzero(flags2>0.1),:]
        partialAfter = pred[np.flatnonzero(flags2>0.1),:]

    return pred, partial, holeAfter, scal

def processShape(model, opt):
    filename = os.path.join(opt.inputFolder, opt.object + '.obj')

    #Load the 3D mesh
    mesh = o3d.io.read_triangle_mesh(filename)

    #Compute mesh properties
    area = mesh.get_surface_area()
    center = mesh.get_center()
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()

    print(f'Area:{area}')
    print(f'Center: {center}')
    print(f'Min. bound: {min_bound}')
    print(f'Max. bound: {max_bound}')
    max_min = np.max(np.abs(min_bound))
    max_max = np.max(np.abs(max_bound))
    scale = 1/max(max_min, max_max)
    vertices = np.asarray(mesh.vertices)

    #Scale the object
    vertices = vertices * scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    print(mesh.get_min_bound())
    print(mesh.get_max_bound())
    mesh.compute_vertex_normals()

    listPartial = []
    listHole = []

    #The neural network is executed eight times to compute a good resoution
    for i in range(0,8):
        #Sample point in the surface
        pcd = mesh.sample_points_poisson_disk(number_of_points=2048, init_factor=5)
        #Run the neural network which produces the prediction, the partial input and the missing geometry
        pred, partial, hole, scal = predict(network, pcd)
        #The resulting predictions are stored in a list
        listPartial.append(partial)
        listHole.append(hole)

    #We go back to numpy arrays
    partial = np.vstack(listPartial)
    hole = np.vstack(listHole)

    #Scale back the objects
    partial = partial*scal
    hole = hole*scal

    rot_x = get_rotation_x(np.deg2rad(90))
    partial = add_rotation_to_pcloud(partial, rot_x)
    hole = add_rotation_to_pcloud(hole, rot_x)

    partial = partial/scale
    hole = hole/scale

    partial = partial.astype(np.float32)
    hole = hole.astype(np.float32)

    print(partial.shape)
    print(hole.shape)

    #The resulting geometry is stored as point clouds in disk
    save_point_cloud(os.path.join(opt.outputFolder, opt.object + '_partial.xyz'), partial)
    save_point_cloud(os.path.join(opt.outputFolder, opt.object + '_hole.xyz'), hole)

    return partial, hole

#Cut the shape source taking into account the information in target
def cutShape(source, target, name, thr):
    #Find the maximum Y coordinate in target shape
    vertTarget = np.asarray(target.vertices)
    vertSource = np.asarray(source.vertices)

    maxY = np.max(vertTarget[:,1])
    minY = np.min(vertTarget[:,1])

    #We set the cut threshold
    threshold = minY + thr*(maxY - minY)

    #We keep only the geometry below the threshold line
    mapping = []
    cont = 0
    newVert = []

    for i in range(vertSource.shape[0]):
        if vertSource[i,1] <= threshold:
            mapping.append(cont)
            cont = cont + 1
            newVert.append(vertSource[i])
        else:
            mapping.append(-1)
    
    newTri = []

    triSource = np.asarray(source.triangles)
    for i in range(triSource.shape[0]):
        if mapping[triSource[i,0]] != -1 and mapping[triSource[i,1]] != -1 and mapping[triSource[i,2]] != -1:
            newTri.append(np.array([mapping[triSource[i,0]], mapping[triSource[i,1]], mapping[triSource[i,2]]]))
    
    newMesh = o3d.geometry.TriangleMesh()
    newMesh.vertices = o3d.utility.Vector3dVector(np.vstack(newVert))
    newMesh.triangles = o3d.utility.Vector3iVector(np.vstack(newTri))
    
    newMesh.compute_vertex_normals()
    normals = np.asarray(newMesh.triangle_normals)
    predominantOrientation = np.mean(normals, axis=0)

    print(f'Predominant orientation: {predominantOrientation}')
    
    newTri = []
    if predominantOrientation[1] > 0.0:
        triSource = np.asarray(newMesh.triangles)
        for i in range(triSource.shape[0]):
            newTri.append(np.array([triSource[i,2], triSource[i,1], triSource[i,0]]))
        newMesh.triangles = o3d.utility.Vector3iVector(np.vstack(newTri))
        newMesh.compute_vertex_normals()

    target.compute_vertex_normals()

    target.paint_uniform_color([0.4,0.4,0.4])
    newMesh.paint_uniform_color([0.95, 0.7, 0.05])
    o3d.visualization.draw_geometries([newMesh, target])

    return newMesh

parser = argparse.ArgumentParser()
parser.add_argument('--object', type=str, default='', help='Name of the OBJ file with object to repair')
parser.add_argument('--inputFolder', type=str, default='', help='Folder with the objects to repair')
parser.add_argument('--model', type=str, default='', help='Name of the neural network')
parser.add_argument('--outputFolder', type=str, default='', help='Folder with the results')
parser.add_argument('--save', action='store_true', help='optional flag to save the result')
parser.add_argument('--ratio', type=float, default=0.1, help='The threshold to remove the geometry')
#parser.add_argument('--nameOutput', type=str, default='', help='')
opt = parser.parse_args()

#Set the CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using cuda device")
    torch.cuda.set_device(device)

#Load the neural network
network = MSNmodel(2048, device).to(device)
network.apply(weights_init)
network.cuda()

print(opt.model, os.path.isfile(opt.model + "/model.pth"))

if opt.model != '' and os.path.isfile(opt.model + "/model.pth"):
    model_checkpoint = torch.load(opt.model + "/model.pth",map_location='cuda:0')
    residual_checkpoint = torch.load(opt.model + "/residual.pth",map_location='cuda:0')
    
    print("Model network weights loaded ")
    network.model.load_state_dict(model_checkpoint['state_dict'])

    
    #print("Residual network weights loaded ")
    network.residual.load_state_dict(residual_checkpoint['state_dict'])

network.model.eval()
network.residual.eval()

#Process the object with the neural network
pcdPartial, pcdHole = processShape(network, opt)

filenamePred = os.path.join(opt.outputFolder, opt.object + '_pred.xyz')
filenameOff = os.path.join(opt.outputFolder, opt.object + '.off')

pcdHole2 = consolidatePointCloud2(pcdPartial, pcdHole)
consolidatePointCloud(pcdPartial, pcdHole2, filenamePred)

command = 'meshlabserver -i ' + filenamePred + ' -o ' + filenameOff + '.off -s ./scripts/reconstruction.mlx'
os.system(command)

filenameOriginal = os.path.join(opt.inputFolder, opt.object + '.obj')
filenameProc = os.path.join(opt.outputFolder, opt.object + '.off.off')

#We read both meshes: the original and the reconstruction    
mesh1 = o3d.io.read_triangle_mesh(filenameOriginal)
mesh2 = o3d.io.read_triangle_mesh(filenameProc)

#We cut the reconstructed shape only to cover the base
meshResult = cutShape(mesh2, mesh1, opt.object, opt.ratio)

#Optionally, we save the result
if opt.save:
    filenameSave = os.path.join(opt.outputFolder, opt.object + '.off')
    o3d.io.write_triangle_mesh(filenameSave, meshResult)
    filenameSaveOrig = os.path.join(opt.outputFolder, opt.object + '_original.off')
    o3d.io.write_triangle_mesh(filenameSaveOrig, mesh1)

