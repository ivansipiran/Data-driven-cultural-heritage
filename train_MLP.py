from dataset.ShapeNetDataset import *
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from losses.emd import emd_module as emd
from losses.chamfer import champfer_loss as chamfer
from models.hole_residual import MSNautoencoder,  PointNetRes, MSNmodel, NormalModel
from utils.utils import weights_init, visdom_show_pc, save_paths, save_model, vis_curve
from utils.metrics import AverageValueMeter
from losses.MDS import MDS_module
import visdom
import sys

#Input options
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--loss', type=str, default = "emd",  help='loss distance')
parser.add_argument('--visualize', type=bool, default = True,  help='bool visualize')
parser.add_argument('--vis_step', type=int, default = 50,  help='visualize step')
parser.add_argument('--vis_step_test', type=int, default = 20,  help='visualize step')
parser.add_argument('--net_alfa', type=float, default = 2000,  help='net loss weight')
parser.add_argument('--vis_port', type=int, default = 8997,  help='visdom_port')
parser.add_argument('--vis_port_test', type=int, default = 8998,  help='visdom_port')
parser.add_argument('--vis_env', type=str, default = "ENV",  help='visdom environment')
parser.add_argument('--gpu_n', type=int, default = 0,  help='cuda gpu device number')
parser.add_argument('--lrate', type=float, default = 0.0005,  help='learning rate')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives')


opt = parser.parse_args()

#We use Visdom to see the training progress
vis = visdom.Visdom(port = opt.vis_port, env= opt.vis_env + " TRAIN")
vis_test = visdom.Visdom(port = opt.vis_port_test , env= opt.vis_env + " TEST")


# initialize variables
dir_name, logname  = save_paths(opt.model, "train_MLP", "ShapeNetDataset", "hole_residual")

rec_loss1_train = AverageValueMeter()
rec_loss1_test = AverageValueMeter()
rec_loss2_train = AverageValueMeter()
rec_loss2_test = AverageValueMeter()
rec_loss_train = AverageValueMeter()
rec_loss_test = AverageValueMeter()


best_loss = 20000

n_models = 10

# Shapenet part dataloader
class_choice = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Guitar': 6, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Skateboard': 14, 'Table': 15}
dataset_dir = './data/shapenet_part'

dataset_train = ShapeNetDataset(root_dir=dataset_dir, class_choice=class_choice, npoints=2048, split='train')
dataloader_train = DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNetDataset(root_dir=dataset_dir, class_choice=class_choice, npoints=2048, split='test')
dataloader_test = DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


device = torch.device("cuda:" + str(opt.gpu_n) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using cuda device")
    torch.cuda.set_device(device)

#Create Ours-MBD model and send it to GPU
network = NormalModel(opt.num_points, device).to(device)
network.apply(weights_init)
network.cuda()


# Create the optimizers
lrate_m = 0.0005
lrate_r = 0.0005
model_optimizer = optim.Adam(network.model.parameters(), lr = lrate_m)
residual_optimizer = optim.Adam(network.residual.parameters(), lr = lrate_r)



#Load a pretrained model to continue training
if opt.model != '' and os.path.isfile("log/" + opt.model + "/model.pth"):
    
    model_checkpoint = torch.load("log/" + opt.model + "/model.pth")
    residual_checkpoint = torch.load("log/" + opt.model + "/residual.pth")
    
    print("Model network weights loaded ")
    network.model.load_state_dict(model_checkpoint['state_dict'])
    model_optimizer.load_state_dict(model_checkpoint['optimizer'])
    
    print("Residual network weights loaded ")
    network.residual.load_state_dict(residual_checkpoint['state_dict'])
    residual_optimizer.load_state_dict(residual_checkpoint['optimizer'])
    

# save model architecture
with open(logname, 'a') as f: #open and append
        f.write(str(network.model) + '\n')
        f.write(str(network.residual)+'\n')


n_points_out = opt.num_points // 2
labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(n_points_out//opt.n_primitives)+1)).view(n_points_out//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

#Main loop for training
for epoch in range(opt.nepoch):
    network.model.train()
    network.residual.train()

    # -----------------------------------training phase ----------------------------------
    for i, data in enumerate(dataloader_train, 0):
        
        # train generator
        model_optimizer.zero_grad()
        residual_optimizer.zero_grad()
        
        name, in_partial, in_hole, in_complete = data
        
        in_partial = in_partial.contiguous().float().to(device)
        in_hole = in_hole.contiguous().float().to(device)
        in_complete = in_complete.contiguous().float().to(device)
        
        output, output2, rec_loss1, rec_loss2, exp_loss = network(in_partial, in_hole, in_complete, 0.005, 50)
        rec_g_loss = rec_loss1 + rec_loss2 + exp_loss
        
        rec_g_loss.backward()
        
        model_optimizer.step() 
        residual_optimizer.step() 
                
        # values to plot and save
        rec_loss_train.update(rec_g_loss.item())
        rec_loss1_train.update(rec_loss1.item())
        rec_loss2_train.update(rec_loss2.item())
        
        
        # visualization 
        if opt.visualize and i % opt.vis_step == 0:
            idx = random.randint(0, in_partial.size()[0] - 1)
            # print(name[idx])
            pc_rec = np.concatenate((in_partial.contiguous()[idx].data.cpu().numpy(), output.contiguous()[idx].data.cpu().numpy()))
            visdom_show_pc(in_hole.contiguous()[idx].data.cpu(), "TRAIN_IN_HOLE", "IN_HOLE", vis)
            visdom_show_pc(in_complete.contiguous()[idx].data.cpu(), "TRAIN_IN_COMPLETE", "TRAIN_IN_COMPLETE", vis)
            visdom_show_pc(in_partial.contiguous()[idx].data.cpu(), "TRAIN_IN_PARTIAL", "TRAIN_IN_PARTIAL", vis)
            visdom_show_pc(output.contiguous()[idx].data.cpu(), "TRAIN_OUT_HOLE", "TRAIN_OUT_HOLE", vis, Y =labels_generated_points[0:output.size(1)] )
            visdom_show_pc(output2.contiguous()[idx].data.cpu(), "TRAIN_OUT_COMPLETE", "TRAIN_OUT_COMPLETE", vis)
            visdom_show_pc(pc_rec, "TRAIN_OUT_MERGE", "TRAIN_OUT_MERGE", vis)
        
        
        # log per batch
        print("train -> E: ", epoch, "/", i, " Loss: ", rec_loss_train.val, " EMD1: ", rec_loss1_train.val, " EMD2: ", rec_loss2_train.val, " penalty: ", exp_loss.item())
        
        
#    ---------------------------------validation phase--------------------------------------
    if epoch % 5 == 0:
        network.model.eval()
        network.residual.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                name, in_partial, in_hole, in_complete = data
        
                in_partial = in_partial.contiguous().float().to(device)
                in_hole = in_hole.contiguous().float().to(device)
                in_complete = in_complete.contiguous().float().to(device)
                
                output, output2, rec_loss1, rec_loss2, exp_loss = network(in_partial, in_hole, in_complete, 0.005, 50)
                rec_g_loss = rec_loss1 + rec_loss2 + exp_loss
                
                # values to plot and save
                rec_loss_test.update(rec_g_loss.item())
                rec_loss1_test.update(rec_loss1.item())
                rec_loss2_test.update(rec_loss2.item())
                                
                # visualization 
                if opt.visualize and i % opt.vis_step == 0:
                    idx = random.randint(0, in_partial.size()[0] - 1)
                    pc_rec = np.concatenate((in_partial.contiguous()[idx].data.cpu().numpy(), output.contiguous()[idx].data.cpu().numpy()))
                    visdom_show_pc(in_complete.contiguous()[idx].data.cpu(), str(i) + " COMPLETE", str(i) + " COMPLETE", vis_test)
                    visdom_show_pc(output.contiguous()[idx].data.cpu(), str(i) + " OUT_HOLE", str(i) + " OUT_HOLE", vis_test, Y =labels_generated_points[0:output.size(1)] )
                    visdom_show_pc(pc_rec, str(i) + " OUT_MERGE", str(i) + " OUT_MERGE", vis_test)
                    visdom_show_pc(output2.contiguous()[idx].data.cpu(), str(i) + " OUT_COMPLETE", str(i) + " OUT_COMPLETE", vis_test)
                
                # log per batch
                print("test -> E: ", epoch, "/", i, " Loss: ", rec_loss_test.val, " EMD1: ", rec_loss1_test.val, " EMD2: ", rec_loss2_test.val, " penalty: ", exp_loss.item())
                
    rec_loss_train.end_epoch()    
    rec_loss_test.end_epoch()    
    rec_loss1_train.end_epoch() 
    rec_loss2_train.end_epoch()
    rec_loss1_test.end_epoch() 
    rec_loss2_test.end_epoch()
    
    
    vis_curve(rec_loss_train.avgs, rec_loss_test.avgs, "rec_loss", "rec_loss", vis)
    vis_curve(rec_loss1_train.avgs, rec_loss1_test.avgs, "rec_loss1", "rec_loss1", vis)
    vis_curve(rec_loss2_train.avgs, rec_loss2_test.avgs, "rec_loss2", "rec_loss2", vis)
    
    
    if (epoch % 5 == 0) or (epoch == opt.nepoch - 1):
        print("Loss reduced from %8.5f to %8.5f" % (best_loss, rec_loss_test.avg))
        best_loss = rec_loss_test.avg
        save_model(network.model.state_dict(), model_optimizer.state_dict(), logname, dir_name, rec_loss_train, rec_loss_test, epoch, lrate_m, rec_loss1_train.avgs, rec_loss1_test.avgs)
        save_model(network.residual.state_dict(), residual_optimizer.state_dict(), logname, dir_name, rec_loss_train, rec_loss_test, epoch, lrate_r, rec_loss2_train.avgs, rec_loss2_test.avgs, net_name= "residual")
    
    
save_model(network.model.state_dict(), model_optimizer.state_dict(), logname, dir_name, rec_loss_train, rec_loss_test, epoch, lrate_m, rec_loss1_train.avgs, rec_loss1_test.avgs,net_name= "model_C")
save_model(network.residual.state_dict(), residual_optimizer.state_dict(), logname, dir_name, rec_loss_train, rec_loss_test, epoch, lrate_r, rec_loss2_train.avgs, rec_loss2_test.avgs, net_name= "residual_C")
