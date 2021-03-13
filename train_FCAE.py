from dataset.ShapeNetDataset import *
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from losses.emd import emd_module as emd
from losses.chamfer import champfer_loss as chamfer
from models.FCAE_model import FCAEmodel
from utils.metrics import AverageValueMeter
from losses.MDS import MDS_module
import visdom
import pickle

import sys

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv2d') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and classname.find('BatchNorm2d') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def visdom_show_pc(pc, window, title, vis, Y = []):
    if len(Y) != 0:
        vis.scatter(X = pc, Y = Y, win = window, opts = 
                dict(
                        title = title,
                        markersize = 2,
                        xtickmin=-1,
		                xtickmax=1,
                        ytickmin=-1,
		                ytickmax=1,
                        ztickmin=-1,
		                ztickmax=1,
                    ),
                )

    else:        
        vis.scatter(X = pc, win = window, opts = 
                    dict(
                            title = title,
                            markersize = 2,
                            xtickmin=-1,
                            xtickmax=1,
                            ytickmin=-1,
                            ytickmax=1,
                            ztickmin=-1,
                            ztickmax=1,
                        ),
                    )

#Create a folder for a model and copy the Python code to produce that model
def save_paths(save_path, trainFile, datasetFile, modelFile):
    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    dir_name =  os.path.join('log', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')
    os.system('cp ./' + trainFile + '.py %s' % dir_name)
    os.system('cp ./dataset/' + datasetFile + '.py %s' % dir_name)
    os.system('cp ./models/' + modelFile + '.py %s' % dir_name)
    
    return dir_name, logname
    
    
#Save important information during training: losses, epochs and pytorch models
def save_model(network_state_dict, optimizer_state_dict, logname, dir_name, train_loss, val_loss, epoch, lrate , loss_avgs_train, loss_avgs_test, net_name = "model"):
    
    with open(dir_name + "/" + net_name + '_loss_avgs_train.pkl','wb') as f: pickle.dump(loss_avgs_train, f)
    with open(dir_name + "/" + net_name + '_loss_avgs_test.pkl','wb') as f: pickle.dump(loss_avgs_test, f)

    log_table = {
      "net" : net_name,
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,

    }
    with open(logname, 'a') as f: 
        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    print('saving net...')

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': network_state_dict,
        'optimizer': optimizer_state_dict
    }
    torch.save(checkpoint, '%s/%s.pth' % (dir_name, net_name))






def vis_curve(train_curve, test_curve, window, name, vis):
    vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(test_curve)))),
                 Y=np.column_stack((np.array(train_curve),np.array(test_curve))),
                 win=window,
                 opts=dict(title=name, legend=[name + "_curve" , name + "_curve" ], markersize=2, ), )


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--model', type=str, default = 'model',  help='optional reload model path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--loss', type=str, default = "emd",  help='loss distance')
parser.add_argument('--visualize', type=bool, default = True,  help='bool visualize')
parser.add_argument('--vis_step', type=int, default = 80,  help='visualize step')
parser.add_argument('--vis_step_test', type=int, default = 30,  help='visualize test step')
parser.add_argument('--net_alfa', type=float, default = 2000,  help='net loss weight')
parser.add_argument('--vis_port', type=int, default = 8997,  help='visdom_port')
parser.add_argument('--vis_port_test', type=int, default = 8998,  help='visdom_port')
parser.add_argument('--vis_env', type=str, default = "ENV",  help='visdom environment')
parser.add_argument('--gpu_n', type=int, default = 0,  help='cuda gpu device number')
parser.add_argument('--lrate', type=float, default = 0.001,  help='learning rate')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives')


opt = parser.parse_args()




vis = visdom.Visdom(port = opt.vis_port, env= opt.vis_env + " TRAIN")
vis_test = visdom.Visdom(port = opt.vis_port_test , env= opt.vis_env + " TEST")



# initialize variables
dir_name, logname  = save_paths(opt.model, "train_FCAE", "ShapeNetDataset", "FCAE_model")

rec_loss_train = AverageValueMeter()
rec_loss_test = AverageValueMeter()


best_loss = 20000
n_models = 10

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


network = FCAEmodel(1024, opt.num_points, opt.num_points).to(device)

network.apply(weights_init)
network.cuda()



# optimizers
lrate = 0.001
model_optimizer = optim.Adam(network.model.parameters(), lr = lrate)



#load model
if opt.model != '' and os.path.isfile("log/" + opt.model + "/model.pth"):
    
    model_checkpoint = torch.load("log/" + opt.model + "/model.pth")
    
    print("Model network weights loaded ")
    network.model.load_state_dict(model_checkpoint['state_dict'])
    model_optimizer.load_state_dict(model_checkpoint['optimizer'])
    
# save model architecture
with open(logname, 'a') as f: #open and append
        f.write(str(network.model) + '\n')


n_points_out = opt.num_points // 2
labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(n_points_out//opt.n_primitives)+1)).view(n_points_out//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

for epoch in range(opt.nepoch):
    network.model.train()
    # -----------------------------------training phase ----------------------------------
    for i, data in enumerate(dataloader_train, 0):
        # train generator
        model_optimizer.zero_grad()
        
        name, in_partial, in_hole, in_complete = data
        
        in_partial = in_partial.contiguous().float().to(device)
        in_hole = in_hole.contiguous().float().to(device)
        in_complete = in_complete.contiguous().float().to(device)
        
        output, rec_loss = network(in_partial, in_complete, 0.005, 50)
             
        
        rec_loss.backward()
        
        model_optimizer.step() 
                
        # values to plot and save
        rec_loss_train.update(rec_loss.item())
        
        # visualization 
        if opt.visualize and i % opt.vis_step == 0:
            idx = random.randint(0, in_partial.size()[0] - 1)
            visdom_show_pc(in_complete.contiguous()[idx].data.cpu(), "TRAIN_IN_COMPLETE", "TRAIN_IN_COMPLETE", vis)
            visdom_show_pc(in_partial.contiguous()[idx].data.cpu(), "TRAIN_IN_PARTIAL", "TRAIN_IN_PARTIAL", vis)
            visdom_show_pc(output.contiguous()[idx].data.cpu(), "TRAIN_OUT_COMPLETE", "TRAIN_OUT_COMPLETE", vis)
        
        
        # log per batch
        print("train -> E: ", epoch, "/", i, 
                " EMD: ", "{:.7f}".format(rec_loss_train.val))
                
        
    
#    ---------------------------------validation phase--------------------------------------
    if epoch % 5 == 0:
        network.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader_test, 0):
                                
                name, in_partial, in_hole, in_complete = data
        
                in_partial = in_partial.contiguous().float().to(device)
                in_hole = in_hole.contiguous().float().to(device)
                in_complete = in_complete.contiguous().float().to(device)
                
                output, rec_loss = network(in_partial, in_complete, 0.005, 50)
                
                # values to plot and save
                rec_loss_test.update(rec_loss.item())
                
                # log per batch
                print("test -> E: ", epoch, "/", i, 
                " EMD: ", "{:.7f}".format(rec_loss_test.val))
                
            
    
    
    rec_loss_train.end_epoch()    
    rec_loss_test.end_epoch()    
    
    
    vis_curve(rec_loss_train.avgs, rec_loss_test.avgs, "rec_loss", "rec_loss", vis)
    
    
    if (epoch % 5 == 0) and (best_loss > rec_loss_test.avg):
        print("Loss reduced from %8.5f to %8.5f" % (best_loss, rec_loss_test.avg))
        best_loss = rec_loss_test.avg
        save_model(network.model.state_dict(), model_optimizer.state_dict(), logname, dir_name, rec_loss_train, rec_loss_test, epoch, lrate, rec_loss_train.avgs, rec_loss_test.avgs)
        

save_model(network.model.state_dict(), model_optimizer.state_dict(), logname, dir_name, rec_loss_train, rec_loss_test, epoch, lrate, rec_loss_train.avgs, rec_loss_test.avgs, net_name= "model_C")