from dataset.ShapeNetDataset import *
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from losses.emd import emd_module as emd
from losses.chamfer import champfer_loss as chamfer
from models.hole_residual import MSNmodel, NormalModel
from utils.utils import weights_init, visdom_show_pc, save_paths, save_model, vis_curve
from utils.metrics import AverageValueMeter
from utils.pcutils import mean_min_square_distance, save_point_cloud
from losses.MDS import MDS_module

import sys

from extensions.chamfer_dist import ChamferDistance

class DevNull:
    def write(self, msg):
        pass

#Only for testing

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--model', type=str, default = 'model',  help='optional reload model path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--loss', type=str, default = "emd",  help='loss distance')
parser.add_argument('--visualize', type=bool, default = True,  help='bool visualize')
parser.add_argument('--vis_step', type=int, default = 30,  help='visualize step')
parser.add_argument('--vis_step_test', type=int, default = 20,  help='visualize test step')
parser.add_argument('--net_alfa', type=float, default = 2000,  help='net loss weight')
parser.add_argument('--vis_port', type=int, default = 8997,  help='visdom_port')
parser.add_argument('--vis_port_test', type=int, default = 8998,  help='visdom_port')
parser.add_argument('--vis_env', type=str, default = "ENV",  help='visdom environment')
parser.add_argument('--gpu_n', type=int, default = 0,  help='cuda gpu device number')
parser.add_argument('--lrate', type=float, default = 0.001,  help='learning rate')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives')
parser.add_argument('--holeSize', type=int, default=35, help='hole size')
parser.add_argument('--outputFolder', type=str, default='', help='Folder output')


opt = parser.parse_args()

# -------------------------------- Load network----------------------------------------
device = torch.device("cuda:" + str(opt.gpu_n) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using cuda device")
    torch.cuda.set_device(device)


network = MSNmodel(opt.num_points, device).to(device)
network.apply(weights_init)
network.cuda()

#load model
print(opt.model, os.path.isfile(opt.model + "/model.pth"))

if opt.model != '' and os.path.isfile(opt.model + "/model.pth"):
    model_checkpoint = torch.load(opt.model + "/model.pth",map_location='cuda:0')
    residual_checkpoint = torch.load(opt.model + "/residual.pth",map_location='cuda:0')
    
    print("Model network weights loaded ")
    network.model.load_state_dict(model_checkpoint['state_dict'])

    network.residual.load_state_dict(residual_checkpoint['state_dict'])

print(f'**************************  Our - {opt.holeSize/100} ***********************************')


# Shapenet
n_models = 13
class_choice = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Guitar': 6, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Skateboard': 14, 'Table': 15}
categories = class_choice.keys()

R = []

chamfer_dist = ChamferDistance()

#Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

for categorie in categories:

    pred_error = AverageValueMeter()
    gt_error = AverageValueMeter()
    chamfer_error = AverageValueMeter()


    dataset_dir = './data/shapenetcore_part'

    dataset_test = ShapeNetDataset(root_dir=dataset_dir, class_choice={categorie}, npoints=2048, split='test', hole_size=opt.holeSize/100)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=0)
    
    network.model.eval()
    network.residual.eval()

    L = []

    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):

            
            name, in_partial, in_hole, in_complete = data

            in_partial = in_partial.contiguous().float().to(device)
            in_hole = in_hole.contiguous().float().to(device)
            in_complete = in_complete.contiguous().float().to(device)
            
            output, output2, rec_loss1, rec_loss2, exp_loss = network(in_partial, in_hole, in_complete, 0.005, 50)
            
            dist = chamfer_dist(output2, in_complete)
            chamfer_error.update(dist.item()*10000)

            pred = output2.cpu().numpy()[0]
            gt = in_complete.cpu().numpy()[0]
            partial = in_partial.cpu().numpy()[0]
            hole = in_hole.cpu().numpy()[0]
            
            pred_error.update(mean_min_square_distance(pred, gt)*10000)
            gt_error.update(mean_min_square_distance(gt, pred)*10000)

            #Save models and metric
            log_table = {"name":name, "chamfer": dist.item()*10000}
            L.append(log_table)
            #print(name)
            save_point_cloud(os.path.join(opt.outputFolder, categorie, name[0]+'_gt.xyz'), gt)
            save_point_cloud(os.path.join(opt.outputFolder, categorie, name[0]+'_partial.xyz'), partial)
            save_point_cloud(os.path.join(opt.outputFolder, categorie, name[0]+'_pred.xyz'), pred)
            save_point_cloud(os.path.join(opt.outputFolder, categorie, name[0]+'_hole.xyz'), hole)

        gt_error.end_epoch() 
        pred_error.end_epoch()
        chamfer_error.end_epoch()
    
    with open(os.path.join(opt.outputFolder, categorie+".txt"), 'w') as fi:
        fi.write(json.dumps(L))
    
    
    R.append({'cat': categorie, 'chamfer': chamfer_error.avg, 'pred': pred_error.avg, 'gt':gt_error.avg})  

print('Categorie:', end='\t')
print('Chamfer:', end='\t')
print('Pred->GT:', end='\t')
print('GT->Pred:', end='\t')
print()

for dc in R:
    print(dc['cat'], end='\t')    
    print(dc['chamfer'], end='\t')
    print(dc['pred'], end='\t')
    print(dc['gt'], end='\t')
    print()
