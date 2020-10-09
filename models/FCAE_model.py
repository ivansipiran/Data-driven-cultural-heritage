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

class FCAEmodel(nn.Module):
    def __init__(self, bottleneck_size, npoints_in, npoints_out):
        super(FCAEmodel, self).__init__()
        self.loss = emd.emdModule()
        self.loss2 = chamfer.ChamferLoss()

        self.model = Autoencoder(bottleneck_size, npoints_in, npoints_out)
        
    def forward(self, in_partial, in_complete, eps, iters):
        out_complete = self.model(in_partial.transpose(1,2).contiguous())
        
        # loss
        dist, _ = self.loss(out_complete, in_complete, eps, iters)
        emd = torch.sqrt(dist).mean(1)
        loss = emd.mean()

        return out_complete, loss