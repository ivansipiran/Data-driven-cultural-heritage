import visdom
import os
import random
import json
import numpy as np
import torch
import pickle
import matplotlib
import matplotlib.pyplot as plt

#initialize the weighs of the network for Convolutional layers and batchnorm layers
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



def generate_training_plot(path, name, train_loss, test_loss, best_train, best_test):
    plt.figure()
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.xlabel("epoch(" + "Best train:" + "{:.4f}".format(best_train) + " Best test:" + "{:.4f}".format(best_test) +")")
    plt.ylabel("loss")
    plt.text(0.5, 3, "text on plot")
    plt.savefig(os.path.join(path, name) + ".png")
    plt.show()


def open_pickle(path):
    print(path)
    file = open(path, "rb")
    obj = pickle.load(file)
    return obj


