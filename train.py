import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from LoadData import ScanNet2d
from Dataset_processing import ComputeMeanofInput
from model import VGGNet, FCN8s
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
import argparse
import util
import trainer
import torch.nn.functional as F
import shutil

def train(cfg):
    n_class    = int(cfg["data"]["n_class"])
    img_h = int(cfg["data"]["img_h"])
    img_w = int(cfg["data"]["img_w"])
    batch_size = int(cfg["training"]["batch_size"])
    epochs     = int(cfg["training"]["epochs"])
    lr         = float(cfg["training"]["optimizer"]["lr"])
    momentum   = float(cfg["training"]["optimizer"]["momentum"])
    w_decay    = float(cfg["training"]["optimizer"]["weight_decay"])
    step_size  = int(cfg["training"]["lr_schedule"]["step_size"])
    gamma      = float(cfg["training"]["lr_schedule"]["gamma"])
    configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}_input_size{}_03091842".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay, img_h)
    print("Configs:", configs)
    
    root_dir = cfg["data"]["root_dir"]
    train_filename = cfg["data"]["train_file"]
    val_filename=cfg["data"]["val_file"]
    mean_filename= cfg["data"]["mean_file"]
    class_weight_filename= cfg["data"]["class_weight_file"]

    train_file = os.path.join(root_dir, train_filename)
    print(train_file)
    val_file = os.path.join(root_dir, val_filename)
    mean_file = os.path.join(root_dir,mean_filename)
    class_weight_file = os.path.join(root_dir, class_weight_filename)
    model_dir= cfg["training"]["model_dir"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, configs)

    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    continue_train = False
    #MeanRGB_train = ComputeMeanofInput(train_file)
    #MeanRGB_train = np.load(mean_file)
    MeanRGB_train = np.array([0.0,0.0,0.0])
    print("MeanRGB_train: {}".format(MeanRGB_train))
    train_data = ScanNet2d(csv_file=train_file, phase = 'train', trainsize=(img_h,img_w), MeanRGB=MeanRGB_train)
    val_data = ScanNet2d(csv_file=val_file, phase = 'val', trainsize=(img_h,img_w), MeanRGB=MeanRGB_train)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)

    #class_weight = trainer.computer_class_weights(train_file)
    class_weight = np.load(class_weight_file)
    print("class_weight: {}".format(class_weight))
    class_weight = torch.from_numpy(class_weight)
    print("shape of class weight {}".format(class_weight.shape))
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCN8s(encoder_net=vgg_model, n_class =n_class)

    if use_gpu:
        ts = time.time()
        vgg_model = vgg_model.cuda()
        fcn_model = fcn_model.cuda()
        fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
        class_weight = class_weight.cuda()
        print("Finish cuda loading, tme elapsed {}".format(time.time() -ts))

    L = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    #optimizer = optim.SGD(fcn_model.parameters(), lr=lr, momentum= momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=gamma)

    score_dir = os.path.join("scores", configs)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    log_headers = ['epoch', 
                    'train/loss', 
                    'train/acc', 
                    'train/acc_cls', 
                    'train/mean_iu', 
                    'train/fwavacc', 
                    'val/loss',
                    'val/acc',
                    'val/acc_cls',
                    'val/mean_iu',
                    'val/fwavacc',
                    'elapsed_time'
                    ]
    if not os.path.exists(os.path.join(score_dir, 'log.csv')):
        with open(os.path.join(score_dir, 'log.csv'), 'w') as f:
            f.write(','.join(log_headers)+'\n')
    
    IU_scores = np.zeros((epochs, n_class+1))
    pixel_scores = np.zeros(epochs)
    writer = SummaryWriter()
   # color_mapping = util.GenerateColorMapping(n_class)
    best_mean_iu = 0
    epoch_loss = 0.0
    if continue_train:
        model_path = "C:\\Users\\ji\\Documents\\FCN-VGG16\\models\\FCNs-BCEWithLogits_batch1_epoch500_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0.0_w_decay1e-05"
        fcn_model= torch.load(model_path)
        fcn_model.train()
    for epoch in range(epochs):
    
        fcn_model.train()
        scheduler.step()
        ts = time.time()
        running_loss = 0.0
        ######
        label_preds = []
        label_trues = []
        ######
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])
            
           
            outputs = fcn_model(inputs)
            #print("out: {}".format(outputs.shape))
            #print("label: {}".format(labels.shape))
            #print(outputs)
            #print(labels)
            loss = L(outputs, labels)
           # print(loss.shape)
            loss = loss.permute(0,2,3,1).reshape(-1, n_class+1)  #.view(-1,n_class+1)
           # print(loss.shape)
            loss = torch.mean(loss, dim=0)
           # print(loss.shape)
            loss = torch.mul(loss,class_weight)
           # print(loss.shape)
            loss = torch.mean(loss)
           # print(loss)
            loss.backward()
            # print("grad")
            # print(fcn_model.outp.weight.grad)
            # print(fcn_model.embs[0].weight.grad)
            optimizer.step()
            #scheduler.step()
            
            if i==0 and epoch==0:
                # count= util.count_parameters(fcn_model)
                # print("number of parameters in model {}".format(count))
                visIn= inputs[:3]
                #print('shape of in {}'.format(visIn[:5].shape))
                visLabel = batch['l'][:3]
            epoch_loss += loss.item()
            running_loss += loss.item()
           # print("loss: {}".format(loss.data))
            if i%10 ==9 and i!= 0:
                print("epoch{}, iter{}, Iterloss: {}".format(epoch, i, running_loss/10))
                writer.add_scalar('train/iter_loss', running_loss/10, epoch*len(train_loader)+i)
                running_loss = 0.0
                # N, _, h, w = outputs.shape
                # targets = batch['l'].cpu().numpy().reshape(N,h,w)
                # outputs = outputs.data.cpu().numpy()
                # preds_v, targets_v = util.visulaize_output(outputs,targets,color_mapping,n_class)
                # writer.add_images('train/predictions',torch.from_numpy(preds_v),dataformats='NHWC')
        
                # writer.add_images('train/targets',torch.from_numpy(targets_v),dataformats='NHWC')
            #####################################
            outputs = outputs.data.cpu().numpy()
            N, _, h, w = outputs.shape
            pred = outputs.transpose(0,2,3,1).reshape(-1, n_class+1).argmax(axis=1).reshape(N,h,w)
            target = batch['l'].cpu().numpy().reshape(N,h,w)
        #########
            for lt, lp in zip(target, pred):
                label_trues.append(lt)
                label_preds.append(lp)
                
        metrics = util.label_accuracy_score(label_trues, label_preds, n_class+1)
        with open(os.path.join(score_dir, "log.csv"), 'a') as f:
            log = [epoch] + [epoch_loss] + list(metrics)+['']*7
            log = map(str, log)
            f.write(','.join(log) + '\n')
        ########################################
        #scheduler.step()

        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        print("Finish epoch{}, epoch loss {}, time eplapsed {}".format(epoch, epoch_loss, time.time() -ts))
        epoch_loss = 0.0
        ####################
        writer.add_scalar('train/mean_iu', metrics[2], epoch)
        writer.add_scalar('train/acc', metrics[0], epoch)
        writer.add_scalar('train/acc_cls', metrics[1], epoch)
        ######################
        #Training precess visulize
        visOut = fcn_model(visIn)
        preds_v,targets_v = util.visulaize_output(visOut,visLabel,n_class)
        writer.add_images('train/predictions',torch.from_numpy(preds_v),global_step=epoch,dataformats='NHWC')
        writer.add_images('train/targets',torch.from_numpy(targets_v),global_step=epoch,dataformats='NHWC')

        if not os.path.exists(model_path):
            os.makedirs(model_path)
               
        torch.save(fcn_model, os.path.join(model_path,str(epoch)))
        best_mean_iu =val_model(epoch,val_loader,fcn_model,use_gpu,n_class,IU_scores,pixel_scores,score_dir,writer,best_mean_iu,model_path, L)
    
    writer.flush()
    writer.close()
# def train_model(epochs,scheduler,train_loader,val_loader,optimizer,fcn_model,L,model_path,use_gpu,n_class,IU_scores,pixel_scores,score_dir,writer):
#     epoch_loss = 0.0
#     for epoch in range(epochs):
#         scheduler.step()
#         ts = time.time()
#         for iter,batch in enumerate(train_loader):
#             optimizer.zero_grad()

#             if use_gpu:
#                 inputs = Variable(batch['X'].cuda())
#                 labels = Variable(batch['Y'].cuda())
#             else:
#                 inputs, labels = Variable(batch['X']), Variable(batch['Y'])

#             outputs = fcn_model(inputs)
#             #print("out: {}".format(outputs.shape))
#             #print("label: {}".format(labels.shape))
#             #print(outputs)
#             #print(labels)
#             loss = L(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#            # print("loss: {}".format(loss.data))
#             if iter %10 ==0:
#                 print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data))
#         writer.add_scalar('train/loss', epoch_loss, epoch)
#         print("Finish epoch{}, loss {}, time eplapsed {}".format(epoch, epoch_loss, time.time() -ts))
#         epoch_loss = 0.0
#         torch.save(fcn_model, model_path)

    
#         val_model(epoch,val_loader,fcn_model,use_gpu,n_class,IU_scores,pixel_scores,score_dir,writer)


def val_model(epoch,val_loader,fcn_model,use_gpu,n_class,IU_scores,pixel_scores,score_dir, writer,best_mean_iu, model_path, L):
    
    fcn_model.eval()
   
    total_ious = []
    total_ious_valid = []
    pixel_accs = []
    val_loss = 0
    #######
    label_trues = []
    label_preds = []
    visualizations = []
    t = time.time()
    ########
    #print("len of val data loader :{}".format(len(val_loader)))
    for n,batch in enumerate(val_loader):
       # print(iter)
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())
        else:
            inputs = Variable(batch['X'])
            labels = Variable(batch['Y'])

        output = fcn_model(inputs)
        #loss = L(output, labels)
        # loss = torch.mean(loss)
        # val_loss += loss.item()
        output = output.data.cpu().numpy()
   
        N, _, h, w = output.shape
        pred = output.transpose(0,2,3,1).reshape(-1, n_class+1).argmax(axis=1).reshape(N,h,w)
       # print(pred.shape)

        target = batch['l'].cpu().numpy().reshape(N,h,w)
        #########
        imgs = inputs.data.cpu().numpy()
        imgs = imgs.transpose(0,2,3,1)
        for img, lt, lp in zip(imgs, target, pred):
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 3:
                viz = util.visulaize_segmentation(lp,lt,img)
                visualizations.append(viz)


        if n==0:
                # count= util.count_parameters(fcn_model)
                # print("number of parameters in model {}".format(count))
            valIn = inputs[:3]
                    #print('shape of in {}'.format(visIn[:5].shape))
            valLabel = batch['l'][:3]
    # save viz
    viz_dir = os.path.join(score_dir, "visualization_viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    [im.save(os.path.join(viz_dir,str(epoch)+"_i"+str(i)+".png")) for i,im in enumerate(visualizations)]
    
    metrics = util.label_accuracy_score(label_trues, label_preds, n_class+1)
    with open(os.path.join(score_dir, "log.csv"), 'a') as f:
        elapsed_time = time.time()-t
        log = [epoch] + [''] * 6 + list(metrics) + [elapsed_time]
        log = map(str, log)
        f.write(','.join(log) + '\n')
    mean_iu = metrics[2]
    is_best = mean_iu> best_mean_iu
    if is_best:
       best_mean_iu = mean_iu
    if not os.path.exists(model_path):
            os.makedirs(model_path)
    torch.save(fcn_model, os.path.join(model_path,str(epoch)))
    if is_best:
        import shutil
        shutil.copy(os.path.join(model_path,str(epoch)), os.path.join(model_path,'model_best'))


   #writer.add_scalar('val/val_loss', val_loss/len(val_loader),epoch)
    writer.add_scalar('val/meanIoU', mean_iu, epoch)
    writer.add_scalar('val/acc', metrics[0], epoch)
    writer.add_scalar('val/acc_cls', metrics[2], epoch)
        #writer.:('val/meanIoUValidClass', np.nanmean(ious_valid), epoch)
    print("epoch{}, val_acc: {}, val_acc_cls: {}, val_mean_iu: {}".format(epoch, metrics[0], metrics[1], metrics[2]))
        # IU_scores[epoch] = ious
        # np.save(os.path.join(score_dir, "meanIU"), IU_scores)
        # pixel_scores[epoch]=pixel_accs
        # np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
        # #Training precess visulize on val data
    valOut = fcn_model(valIn)
    preds_val,targets_val = util.visulaize_output(valOut,valLabel,n_class)
    writer.add_images('val/predictions',torch.from_numpy(preds_val),global_step=epoch,dataformats='NHWC')
    writer.add_images('val/targets',torch.from_numpy(targets_val),global_step=epoch,dataformats='NHWC')

    return best_mean_iu
def iou(pred, target,n_class):
    ious = []
    for cls in range(0,n_class+1):
        pred_inds = pred ==cls
        target_inds = target ==cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection)/max(union, 1))
    return ious
def iou_valid(pred, target,n_class):
    ious = []
    for cls in range(0,n_class):
        pred_inds = pred ==cls
        target_inds = target ==cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection)/max(union, 1))
    return ious

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct/ total

# def cross_entropy2d(input, target, weight=None, size_average=True):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()
#     # log_p: (n, c, h, w)
#     if LooseVersion(torch.__version__) < LooseVersion('0.3'):
#         # ==0.2.X
#         log_p = F.log_softmax(input)
#     else:
#         # >=0.3
#         log_p = F.log_softmax(input, dim=1)
#     # log_p: (n*h*w, c)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
#     log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p.view(-1, c)
#     # target: (n*h*w,)
#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
#     if size_average:
#         loss /= mask.data.sum()
#     return loss


if __name__=="__main__":
  #  val(0)
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default=".\\FCN8s_VGG16.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    train(cfg)








