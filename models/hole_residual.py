import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append("..")
from losses.expansion_penalty import expansion_penalty_module as expansion
from losses.MDS import MDS_module
from losses.emd import emd_module as emd
from losses.chamfer import champfer_loss as chamfer

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint,RAN = True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if RAN :
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

class PointNetfeat(nn.Module):
    def __init__(self, num_points = 8192, global_feat = True):
        super(PointNetfeat, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh() 

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x)) * 0.15
        return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 8192):
        super(PointGenCon, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


# input feature vector should have shape (b, m)
# b = batch, m = number of features, n = number of points
# output pointcloud has shape (b, n, 3)
class MSNdecoder(nn.Module):
    def __init__(self, num_points = 8192, bottleneck_size = 1024, n_primitives = 16):
        super(MSNdecoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.n_primitives)])
        self.expansion = expansion.expansionPenaltyModule()
    
    def forward(self, x):
        outs = []
        
        for i in range(0,self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.n_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        outs = torch.cat(outs,2).contiguous() 
        out1 = outs.transpose(1, 2).contiguous() 
        
        dist, _, mean_mst_dis = self.expansion(out1, self.num_points//self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)
        
        return out1, loss_mst
    
    
    

# input pointcloud should have shape (b, 3, n)
# b = batch, m = number of features, n = number of points
# output pointcloud has shape (b, n, 3)
class NormalDecoder(nn.Module):
    def __init__(self, botleneck_size = 256, npoints = 2048):
        super(NormalDecoder, self).__init__()
        self.npoints = npoints

        self.model = nn.Sequential(
            nn.Linear(in_features= botleneck_size, out_features=1024),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=npoints * 3),
            
        )

    def forward(self, z):
        output = self.model(z.squeeze())
        output = output.view(-1, self.npoints, 3)
        return output



class Autoencoder(nn.Module):
    def __init__(self, bottleneck_size, npoints_in, npoints_out):
        super(Autoencoder, self).__init__()
        self.E = nn.Sequential(
            PointNetfeat(npoints_in, global_feat=True),
            nn.Linear(1024, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU()
        )
        self.D = NormalDecoder(bottleneck_size, npoints_out)
    def forward(self, x):    
        return self.D(self.E(x))


class MSNautoencoder(nn.Module):
    def __init__(self, bottleneck_size, npoints_in, npoints_out):
        super(MSNautoencoder, self).__init__()
        self.E = nn.Sequential(
            PointNetfeat(npoints_in, global_feat=True),
            nn.Linear(1024, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU()
        )
        self.D = MSNdecoder(num_points = npoints_out, bottleneck_size = bottleneck_size, n_primitives = 8)
    def forward(self, x):    
        return self.D(self.E(x))

class MSNmodel(nn.Module):
    def __init__(self, num_points, device):
        super(MSNmodel, self).__init__()
        self.model = MSNautoencoder(1024, num_points, num_points //2)
        self.residual = PointNetRes()
        self.loss = emd.emdModule()
        self.num_points = num_points
        self.device = device

    def forward(self, in_partial, in_hole, in_complete, eps, iters):
        in_partial = in_partial.transpose(2,1).contiguous()
        out_hole, exp_loss = self.model(in_partial)

        # input shape of both pc1, and pc2 should be (b, 3, n)
        def merge_pointClouds(pc1, pc2):
            id1 = torch.zeros(pc1.shape[0], 1, pc1.shape[2]).cuda().contiguous()
            pc1 = torch.cat( (pc1, id1), 1)
            
            id2 = torch.ones(pc2.shape[0], 1, pc2.shape[2]).cuda().contiguous()
            pc2 = torch.cat( (pc2, id2), 1)

            MPC = torch.cat([pc1, pc2], 2).contiguous()
            return MPC
        

        MPC = merge_pointClouds(out_hole.detach().transpose(1, 2).contiguous(), in_partial)
        idx = farthest_point_sample(MPC[:, 0:3, :].transpose(1,2).contiguous(), self.num_points, RAN=False)
        SPC = index_points(MPC.transpose(1,2).contiguous(), idx)
        SPC = SPC.transpose(1,2).contiguous()
        SPC2 = SPC.clone().detach() 


        shift = self.residual(SPC)
        
        flags = SPC[:,3,:]
        flags2 = flags.clone().detach()

        SPC = SPC[:, 0:3, :]
        
        out_complete = (SPC + shift).transpose(2,1).contiguous()
        
        # loss
        dist, _ = self.loss(out_hole, in_hole, eps, iters)
        emd1 = torch.sqrt(dist).mean(1)
        loss1 = emd1.mean()
        
        dist2, _ = self.loss(out_complete, in_complete, eps, iters)
        emd2 = torch.sqrt(dist2).mean(1)
        loss2 = emd2.mean()
        
        return out_hole, out_complete, loss1, loss2, exp_loss, SPC2[:,0:3,:].transpose(2,1).contiguous(), flags2
        

class NormalModel(nn.Module):
    def __init__(self, num_points, device):
        super(NormalModel, self).__init__()
        self.model = Autoencoder(1024, num_points, num_points // 2)
        self.residual = PointNetRes()
        self.loss = emd.emdModule()
        self.loss2 = chamfer.ChamferLoss()
        self.num_points = num_points
        self.device = device

    def forward(self, in_partial, in_hole, in_complete, eps, iters):
        in_partial = in_partial.transpose(2,1).contiguous()
        out_hole = self.model(in_partial)
        
        def merge_pointClouds(pc1, pc2):
            id1 = torch.zeros(pc1.shape[0], 1, pc1.shape[2]).cuda().contiguous()
            pc1 = torch.cat( (pc1, id1), 1)
            
            id2 = torch.ones(pc2.shape[0], 1, pc2.shape[2]).cuda().contiguous()
            pc2 = torch.cat( (pc2, id2), 1)

            MPC = torch.cat([pc1, pc2], 2).contiguous()
            return MPC
        

        MPC = merge_pointClouds(out_hole.detach().transpose(1, 2).contiguous(), in_partial)
        idx = farthest_point_sample(MPC[:, 0:3, :].transpose(1,2).contiguous(), self.num_points, RAN=False)
        SPC = index_points(MPC.transpose(1,2).contiguous(), idx)
        SPC = SPC.transpose(1,2).contiguous()
        
        shift = self.residual(SPC)
        
        SPC = SPC[:, 0:3, :]

        out_complete = (SPC + shift).transpose(2,1).contiguous()
    
    	# Completer loss
        dist, _ = self.loss(out_hole, in_hole, eps, iters)
        emd1 = torch.sqrt(dist).mean(1)
        loss1 = emd1.mean()
        
        dist2, _ = self.loss(out_complete, in_complete, eps, iters)
        emd2 = torch.sqrt(dist2).mean(1)
        loss2 = emd2.mean()

        
        return out_hole, out_complete , loss1, loss2, torch.tensor(0).float().to(self.device)

    
