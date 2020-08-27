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
    configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}_input_size{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay, img_h)
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
    MeanRGB_train = np.load(mean_file)
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
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=gamma)

    score_dir = os.path.join("scores", configs)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    IU_scores = np.zeros((epochs, n_class+1))
    pixel_scores = np.zeros(epochs)
    writer = SummaryWriter()
   # color_mapping = util.GenerateColorMapping(n_class)
    epoch_loss = 0.0
    if continue_train:
        model_path = "C:\\Users\\ji\\Documents\\FCN-VGG16\\models\\FCNs-BCEWithLogits_batch1_epoch500_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0.0_w_decay1e-05"
        fcn_model= torch.load(model_path)
        fcn_model.train()
    for epoch in range(epochs):
       # scheduler.step()
        ts = time.time()
        running_loss = 0.0
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
            optimizer.step()
            scheduler.step()
            
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
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        print("Finish epoch{}, epoch loss {}, time eplapsed {}".format(epoch, epoch_loss, time.time() -ts))
        epoch_loss = 0.0

        #Training precess visulize
        visOut = fcn_model(visIn)
        preds_v,targets_v = util.visulaize_output(visOut,visLabel,n_class)
        writer.add_images('train/predictions',torch.from_numpy(preds_v),global_step=epoch,dataformats='NHWC')
        writer.add_images('train/targets',torch.from_numpy(targets_v),global_step=epoch,dataformats='NHWC')


        torch.save(fcn_model, model_path)
        val_model(epoch,val_loader,fcn_model,use_gpu,n_class,IU_scores,pixel_scores,score_dir,writer)
    
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


def val_model(epoch,val_loader,fcn_model,use_gpu,n_class,IU_scores,pixel_scores,score_dir, writer):
    
    fcn_model.eval()
   
    total_ious = []
    total_ious_valid = []
    pixel_accs = []
    #print("len of val data loader :{}".format(len(val_loader)))
    for n,batch in enumerate(val_loader):
       # print(iter)
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0,2,3,1).reshape(-1, n_class+1).argmax(axis=1).reshape(N,h,w)
       # print(pred.shape)

        target = batch['l'].cpu().numpy().reshape(N,h,w)
        for p,t in zip(pred,target):
            total_ious.append(iou(p,t,n_class))
            total_ious_valid.append(iou_valid(p,t,n_class))
            pixel_accs.append(pixel_acc(p,t))
        
        if n==0:
                # count= util.count_parameters(fcn_model)
                # print("number of parameters in model {}".format(count))
            valIn= inputs[:3]
                #print('shape of in {}'.format(visIn[:5].shape))
            valLabel = batch['l'][:3]
    # preds_v, targets_v = util.visulaize_output(outputs,targets,color_mapping,n_class)
    # writer.add_images('train/predictions',torch.from_numpy(preds_v),dataformats='NHWC')
    # writer.add_images('train/targets',torch.from_numpy(targets_v),dataformats='NHWC')
    total_ious = np.array(total_ious).T
    ious = np.nanmean(total_ious, axis=1)
    total_ious_valid = np.array(total_ious_valid).T
    ious_valid = np.nanmean(total_ious_valid, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    writer.add_scalar('val/pixel_acc', pixel_accs, epoch)
    writer.add_scalar('val/meanIoU', np.nanmean(ious), epoch)
    writer.add_scalar('val/meanIoUValidClass', np.nanmean(ious_valid), epoch)
    print("epoch{}, pix_acc: {}, meanIoU: {}, meanIoU_V: {}".format(epoch, pixel_accs, np.nanmean(ious), np.nanmean(ious_valid)))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch]=pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)
    #Training precess visulize on val data
    valOut = fcn_model(valIn)
    preds_val,targets_val = util.visulaize_output(valOut,valLabel,n_class)
    writer.add_images('val/predictions',torch.from_numpy(preds_val),global_step=epoch,dataformats='NHWC')
    writer.add_images('val/targets',torch.from_numpy(targets_val),global_step=epoch,dataformats='NHWC')

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








