import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
sys.path.append("/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN")
from imports.ABIDEDatasetCryst import ABIDEDatasetCryst
from imports.ABIDEDatasetTotalComp import ABIDEDatasetTotalComp
from net.dynamicbrainwithGMT import CustomNetworkWithGMT
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import scipy.stats as stats
from imports.ABIDEDataset import ABIDEDataset
from torch_geometric.data import DataLoader
from net.braingnn import Network
from net.dynamicbraingnn import CustomNetwork
from imports.utils import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, explained_variance_score
from net.dynamicorthogonalnet import CustomNetworkWithOrthogonality
# import EarlyStopping
from pytorchtools  import EarlyStopping
import random
from sklearn.metrics._regression import r2_score
import wandb
from sklearn import preprocessing
from net.garo.dynamicbraingnnginadd import CustomNetworkWithGINAddGARO
torch.manual_seed(123)


EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)
os.environ['WANDB_DISABLE_CODE'] = 'false'
#wandb.init(project='abcd_fluid_53nodes', save_code=True)
wandb.init(project='results_abcd_gin_topk_add', save_code=True, config="/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN/wandb_sweeps/default_test_sweep.yaml")
#wandb.init(project='abcd_fluid_prediction_publish', save_code=True)
config = wandb.config
torch.manual_seed(1)
np.random.seed(1111)
random.seed(1111)
torch.cuda.manual_seed_all(1111)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN/data/data/Output', help='root directory of the dataset')
parser.add_argument('--stepsize', type=int, default=30, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.1, help='scheduler shrinking rate')
parser.add_argument('--indim', type=int, default=53, help='feature dim')
parser.add_argument('--nroi', type=int, default=53, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=1, help='num of classes')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')

#Arguments from WANDB Sweeps

parser.add_argument('--lr', type = float, default=config.lr, help='learning rate')
parser.add_argument('--epoch', type=int, default=config.epoch, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=config.n_epochs, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=config.batchSize, help='size of the batches')
parser.add_argument('--weightdecay', type=float, default=config.weightdecay, help='regularization')
parser.add_argument('--lamb0', type=float, default=config.lamb0, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=config.lamb1, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=config.lamb2, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=config.lamb3, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=config.lamb4, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=config.lamb5, help='s1 consistence regularization')
parser.add_argument('--reg', type=float, default=config.reg, help='GMT reg')
parser.add_argument('--layer', type=int, default=config.layer, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=config.ratio, help='pooling ratio')
parser.add_argument('--optim', type=str, default=config.optim, help='optimization method: SGD, Adam')
parser.add_argument('--n_layers', type=str, default=config.n_layers, help='Dimensions of hidden layers')
parser.add_argument('--n_fc_layers', type=str, default=config.n_fc_layers, help='Dimensions of fully connected layers')
parser.add_argument('--n_clustered_communities', type=int, default=config.n_clustered_communities, help='Number of clustered communities')
parser.add_argument('--early_stop_steps', type=int, default=config.early_stop_steps, help='Early Stopping Steps')

opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
name = 'ABCD'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))


#Convert Number of Layers into Int List
#FOR JOB
opt.n_layers = [int(float(numeric_string)) for numeric_string in opt.n_layers.split(',')]
opt.n_fc_layers = [int(float(numeric_string)) for numeric_string in opt.n_fc_layers.split(',')]

# #FOR LOCAL
# opt.n_layers = [int(float(numeric_string)) for numeric_string in str(opt.n_layers[0]).split(',')]
# opt.n_fc_layers = [int(float(numeric_string)) for numeric_string in str(opt.n_fc_layers[0]).split(',')]

################## Define Dataloader ##################################

dataset = ABIDEDataset(path,name)
dataset.data.y = dataset.data.y.squeeze()

dataset.data.x[dataset.data.x == float('inf')] = 0

tr_index,val_index,te_index = train_val_test_split(fold=fold, n_subjects=int(dataset.data.x.shape[0]/opt.nroi))
train_dataset = dataset[tr_index]
val_dataset = dataset[val_index]
test_dataset = dataset[te_index]


train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, drop_last=True)


# Define learning rate schedule
def lr_schedule(epoch):
    totalepochs = 100
    if epoch < totalepochs * 0.2:
        lr = 0.0005 + (0.0005 / (totalepochs * 0.2)) * epoch
    else:
        lr = 0.005 - ((0.005 - 5.0e-7) / (totalepochs * 0.8)) * (epoch - totalepochs * 0.2)
    return lr


############### Define Graph Deep Learning Network ##########################
model = CustomNetworkWithGINAddGARO(opt.indim,opt.ratio,opt.nclass,opt.n_layers,opt.n_fc_layers,opt.n_clustered_communities,opt.nroi,opt.reg).to(device)
#model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
print(model)
print("Learning rate is " + str(opt.lr) + "Reduction after 200 epochs"+ " Regularization is "+ str(opt.weightdecay) + "Scheduler is "+ "NO" )



if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.8, weight_decay=opt.weightdecay, nesterov = True)

#scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=15,
                                                               verbose=True)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')

    # for param_group in optimizer.param_groups:
    #     print("LR", param_group['lr'])

    #lr = lr_schedule(epoch)
    for param_group in optimizer.param_groups:
                #param_group['lr'] = lr
                print("LR", param_group['lr'])
        # if epoch > 200:
        #     param_group['lr']=0.001

    model.train()
    s1_list = []
    s2_list = []

    loss_all = 0
    step = 0
    for data in train_loader:
        scores_list=[]
        loss_tpks = []
        loss_pools=[]
        pool_weights =[]
        s=[]
        criterion = torch.nn.SmoothL1Loss()
    
        data = data.to(device)
        optimizer.zero_grad()
        output,allpools, scores = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)

        for i in range(len(scores)):
            scores_list.append(torch.sigmoid(scores[i]).view(output.size(0),-1).view(-1).detach().cpu().numpy())
            s.append(torch.sigmoid(scores[0]).view(output.size(0),-1))
            pool_weights.append(allpools[i].weight)
            loss_pools.append((torch.norm(pool_weights[i], p=2)-1) ** 2)
            loss_tpks.append(topk_loss(s[i],opt.ratio))
        # s1_list.append(s1.view(-1).detach().cpu().numpy())
        # s2_list.append(s2.view(-1).detach().cpu().numpy())
        data.y = data.y.to(torch.float32)

        loss_c = criterion(output.squeeze(), data.y)

        # loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        # loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        # loss_tpk1 = topk_loss(s1,opt.ratio)
        # loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s[0][data.y == c])

        loss = opt.lamb0*loss_c + opt.lamb1 * loss_pools[0] + opt.lamb2 * loss_pools[1] \
                   + opt.lamb3 * loss_tpks[0] + opt.lamb4 *loss_tpks[1] + opt.lamb5* loss_consist
        writer.add_scalar('train/prediction_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss1', loss_pools[0], epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss2', loss_pools[1], epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpks[0], epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpks[1], epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(scores_list[0])
        s2_arr = np.hstack(scores_list[1])


    return loss_all / len(train_dataset), s1_arr, s2_arr ,pool_weights[0],pool_weights[1]


#Regression Metrics

def return_regressor_metrics(labels, pred_prob, loss_value=None):

    print('First 5 values:', labels.shape, labels[:5], pred_prob.shape, pred_prob[:5])
    r2 = r2_score(labels, pred_prob)
    r = stats.pearsonr(labels, pred_prob)[0]
    mse = mean_squared_error(labels, pred_prob)
    mae = mean_absolute_error(labels, pred_prob)

    return {'loss': loss_value,
            'r2': r2,
            'r': r,
            'mse': mse,
            'mae': mae
            }



###################### Network Testing Function#####################################

def test_mse_score(loader):
        model.eval()
        mse_sum = 0
        for data in loader:
            data = data.to(device)
            outputs,allpools, scores= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
            pred = outputs
            pred = torch.nan_to_num(pred)
            mse_sum += mean_squared_error(data.y.cpu().detach().numpy(), pred.cpu().detach().numpy())

        return mse_sum / len(loader)

def test_r2(loader):
        model.eval()
        r2_sum = 0
        for data in loader:
            data = data.to(device)
            outputs, allpools, scores= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
            pred = outputs
            pred = torch.nan_to_num(pred)
            #correct += pred.eq(data.y).sum().item()
            r2_sum += r2_score(data.y.cpu().detach().numpy(), pred.cpu().detach().numpy())

        return r2_sum / len(loader)


def evaluate_model(model, loader, device):

    model.eval()
    predictions = []
    labels = []
    test_error = 0

    criterion = torch.nn.SmoothL1Loss()

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            output_batch,allpools, scores = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
            #output_batch = model(data)
            #output_batch = output_batch.clamp(0, 1)  # For NaNs
            output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch)  # For NaNs
            # output_batch = output_batch.flatten()
            loss = criterion(output_batch, data.y.unsqueeze(1))

            test_error += loss.item() * data.num_graphs
            pred = output_batch.flatten().detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return return_regressor_metrics(labels, predictions,
                                        loss_value=test_error / len(loader.dataset))


def sailency_maps_generator(model, loader, device):

    model.eval()
    allgrads=[]

    for data in loader:
            data = data.to(device)
            data.x.requires_grad_()
            output_batch,allpools, scores = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
            output_batch = torch.where(torch.isnan(output_batch), torch.zeros_like(output_batch), output_batch)  # For NaNs
            grads = torch.autograd.grad(output_batch, data.x, grad_outputs=torch.ones_like(output_batch), create_graph=True)[0]
            allgrads.append(grads.detach().cpu().numpy())

            
    sailent_gradients = sum(allgrads[:-1])
    return sailent_gradients

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data in loader:
        scores_list=[]
        loss_tpks = []
        loss_pools=[]
        pool_weights =[]
        s=[]
        data = data.to(device)
        output, allpools, scores= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)

        for i in range(len(scores)):
            scores_list.append(torch.sigmoid(scores[i]).view(output.size(0),-1).view(-1).detach().cpu().numpy())
            s.append(torch.sigmoid(scores[0]).view(output.size(0),-1))
            pool_weights.append(allpools[i].weight)
            loss_pools.append((torch.norm(pool_weights[i], p=2)-1) ** 2)
            loss_tpks.append(topk_loss(s[i],opt.ratio))

        loss_c = F.mse_loss(output.squeeze(), data.y)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s[0][data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_pools[0] + opt.lamb2 * loss_pools[1] \
                   + opt.lamb3 * loss_tpks[0] + opt.lamb4 *loss_tpks[1] + opt.lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


#######################################################################################
############################   Model Training #########################################
#######################################################################################

# #-----TESTING
# model.load_state_dict(torch.load("/data/users2/bthapaliya/IntelligenceABCDPredictionBrainGNN/model/4test_final_new.pth"))
# model.eval()
# test_metrics=evaluate_model(model, test_loader,device)
# print('Test Stats: MSE: ' + str(test_metrics["mse"]) + " MAE:" + str(test_metrics["mae"]) + " Loss: "+ str(test_metrics["loss"]) + " R: "+ str(test_metrics["r"]) + " R2:" + str(test_metrics["r2"]) )
# log_dict = {'test_r': str(test_metrics['r']), 'test_loss': str(test_metrics['loss']), 'test_mse': str(test_metrics["mse"]), 'test_mae': str(test_metrics["mae"])}
# wandb.log(log_dict)

# #TESTING ENDS

# #sailency maps

# # allgrads = sailency_maps_generator(model, test_loader, device)

# # reshaped_allgrads = allgrads.reshape(opt.batchSize, 100,100)

# # summed_grads = np.abs(reshaped_allgrads.sum(axis=0))

# # #summed_grads.save("FinalSailentGrads.npy", allow_pickle=True)
# # np.save("FinalSailentGrads.npy",summed_grads)
# # sys.exit()

early_stopping = EarlyStopping(patience=24, verbose=True)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
for epoch in range(0, num_epoch):
    since  = time.time()
    tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)

    train_metrics = evaluate_model(model, train_loader,device)
    val_metrics = evaluate_model(model, val_loader, device)
    #scheduler.step()
    print(
            'Epoch: {:03d}, Loss: {:.7f} / {:.7f}, R2: {:.4f} / {:.4f}, R: {:.4f} / {:.4f}, MSE: {:.4f} / {:.4f}, MAE: {:.4f} / {:.4f}'
            ''.format(epoch, train_metrics['loss'], val_metrics['loss'],
                      train_metrics['r2'], val_metrics['r2'],
                      train_metrics['r'], val_metrics['r'],
                      train_metrics['mse'], val_metrics['mse'],
                      train_metrics['mae'], val_metrics['mae']))
    
    log_dict = {
            f'train_loss': train_metrics['loss'], f'val_loss': val_metrics['loss'],
            f'train_r2': train_metrics['r2'], f'val_r2': val_metrics['r2'],
            f'train_r': train_metrics['r'], f'val_r': val_metrics['r'],
            f'train_mse': train_metrics['mse'], f'val_mse': val_metrics['mse'],
            f'train_mae': train_metrics['mae'], f'val_mae': val_metrics['mae']
        }
    # tr_mse = test_mse_score(train_loader)
    # val_mse = test_mse_score(val_loader)
    # val_loss = test_loss(val_loader,epoch)
    val_loss = val_metrics["loss"]
    scheduler.step(val_loss)
    #scheduler.step()

    early_stopping(val_loss, model)
        
    if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print('*====**')
    #print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Epoch: {:03d}, Train Loss: {:.7f}, '
    #       'Train MSE: {:.7f}, Test Loss: {:.7f}, Test MSE: {:.7f}'.format(epoch, tr_loss,
    #                                                    tr_mse, val_loss, val_mse))

    #writer.add_scalars('MSE',{'train_mse':tr_mse,'val_acc':val_mse},  epoch)
    writer.add_scalars('Loss', {'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss']},  epoch)

    # wandb.log({
    #     f'train_mse': tr_mse, f'val_mse': val_mse,
    #     f'train_loss': tr_loss, f'val_loss': val_loss
    # })
    wandb.log(log_dict)
    #writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
    #writer.add_histogram('Hist/hist_s2', s2_arr, epoch)

    if val_loss < best_loss and epoch > 5:
        print("saving best model")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        if save_model:
            torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'test_final_new'+'.pth'))




######################################################################################
######################## Testing on testing set ######################################
######################################################################################


model.load_state_dict(best_model_wts)
model.eval()
test_metrics=evaluate_model(model, test_loader,device)
print('Test Stats: MSE: ' + str(test_metrics["mse"]) + " MAE:" + str(test_metrics["mae"]) + " Loss: "+ str(test_metrics["loss"]) + " R: "+ str(test_metrics["r"]) + " R2:" + str(test_metrics["r2"]) )
log_dict = {'test_r': str(test_metrics['r']), 'test_loss': str(test_metrics['loss']), 'test_mse': str(test_metrics["mse"]), 'test_mae': str(test_metrics["mae"])}
wandb.log(log_dict)