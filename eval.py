import torch
from torch.utils.data import DataLoader
import os
from LoadData import ScanNet2d
from torch.autograd import Variable
import numpy as np
import util

def main():
    path = "C:\\Users\\ji\\Documents\\FCN-VGG16\\models\\FCNs-BCEWithLogits_batch1_epoch10000_RMSprop_scheduler-step50-gamma0.5_lr0.001_momentum0.5_w_decay1e-05_input_size484"
    model = torch.load(path)
    model.eval()
    print("num para") 
    print(util.count_parameters(model))
    root_dir=".\\"
    train_file = os.path.join(root_dir, "train_one.csv")
    train_data = ScanNet2d(csv_file=train_file, phase = 'train', MeanRGB=np.array([0.0,0.0,0.0]))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    n_class = 40
    use_gpu=True
    total_ious = []
    pixel_accs = []
    #print("len of val data loader :{}".format(len(val_loader)))
    for iter,batch in enumerate(train_loader):
        # print(iter)
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0,2,3,1).reshape(-1, n_class+1).argmax(axis=1).reshape(N,h,w)
        # print(pred.shape)

        target = batch['l'].cpu().numpy().reshape(N,h,w)
        for p,t in zip(pred,target):
            ious,count,ds=iou(p,t,n_class)
            total_ious.append(ious)
            pixel_accs.append(pixel_acc(p,t))
    # preds_v, targets_v = util.visulaize_output(outputs,targets,color_mapping,n_class)
    # writer.add_images('train/predictions',torch.from_numpy(preds_v),dataformats='NHWC')
    # writer.add_images('train/targets',torch.from_numpy(targets_v),dataformats='NHWC')
    total_ious = np.array(total_ious).T
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    meanIoU = np.nanmean(ious)
    
    print(total_ious)
    #print(ious)
    print(count)
    print(ds)
    print(np.nansum(ious)/count)
    print("pix_acc: {}, meanIoU: {}".format(pixel_accs, np.nanmean(ious)))

def iou(pred, target,n_class):
    ious = []
    count = 0
    ds = 0
    for cls in range(0,n_class+1):
        pred_inds = pred ==cls
        target_inds = target ==cls
        if target_inds.sum() != 0:
           # print(cls)
            count+=1 
        else:
            ds+=1
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        # if cls==7:
        #     print("yyyy")
        #     print(intersection,union)
        # if target_inds.sum() != 0 and pred_inds.sum() ==0:
        #     print("jjjjj")
        #     print(cls)
        # if target_inds.sum() == 0 and pred_inds.sum() !=0:
        #     print("kkkk")
        #     print(cls)
        if union == 0:
            ious.append(float('nan'))
        else:
           # print(cls)
            ious.append(float(intersection)/max(union, 1))
    return ious,count,ds

def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    print("correct")
    print(correct)
    print("target")
    print(target)

    return correct/ total
if __name__=="__main__":
    main()
