import torch 
import pandas as pd
import numpy as np
import os
from matplotlib.image import imread
import imageio
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

from Dataset_processing import ComputeMeanofInput
from matplotlib import pyplot as plt
import util


num_class = 20

h,w = 968, 1296



class ScanNet2d(Dataset):
    def __init__(self, csv_file, phase, MeanRGB, label_map_file="C:\\Users\\ji\\Documents\\ScanNet-master\\data\\scannetv2-labels.combined.tsv" , trainsize=(int(h/2), int(w/2)), n_class=num_class, crop=False, data_analysis= False):
        self.data = pd.read_csv(csv_file)
        self.means = np.array([0,0,0])
        self.n_class = n_class
        self.label_map_file = label_map_file
        self.crop = crop
        self.phase=phase
        self.valid_class_mapping = util.valid_class_map()
        self.data_analysis= data_analysis
        if self.phase=='train' or self.phase=='val':
            self.crop = True
            self.new_h = trainsize[0]
            self.new_w = trainsize[1]

       # MeanRGB = ComputeMeanofInput(csv_file)
        self.MeanRGB= MeanRGB
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_file=self.data.iloc[idx,1]
        label_file = self.data.iloc[idx,2]
        img  = imread(im_file)[:,:,:3]
        la_ = np.array(imageio.imread(label_file))
        if self.data_analysis:
            label = util.label_mapping(la_, self.label_map_file)
        else:
            label_ = util.label_mapping(la_, self.label_map_file)
            label = util.convert_to_valid_trainId(label_,self.valid_class_mapping)

         
        #print(label)
        # if np.amax(label)>40:
        #     print(idx)
        #     print(np.amax(label))
        #     print(self.data.iloc[idx,1])

        if self.crop:
            h,w,_ = img.shape
            if self.phase=='train':
                top = random.randint(0, h-self.new_h)
                left = random.randint(0, w-self.new_w)
                img = img[top:top+self.new_h, left:left+self.new_w]
                label = label[0:0+self.new_h, 0:0+self.new_w]
            if self.phase=='val':
                img = img[0:0+self.new_h, 0:0+self.new_w]
                label = label[0:0+self.new_h, 0:0+self.new_w]
        #reduce RGB mean
        img = np.transpose(img, (2,0,1))
        img[0] -= self.MeanRGB[0]
        img[1] -= self.MeanRGB[1]
        img[2] -= self.MeanRGB[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # one-hot label
        h, w = label.size()
        target = torch.zeros(self.n_class+1,h,w)
        for c in range(self.n_class+1):
            target[c][label==(c)] =1
        sample = {'X':img,'Y': target, 'l':label}
        return sample

def showbatch(batch):
    img_batch=batch['X']
    img_batch[:,0,...].add_(MeanRGB[0])
    img_batch[:,1,...].add_(MeanRGB[1])
    img_batch[:,2,...].add_(MeanRGB[2])
    BS=len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1,2,0)))

    plt.title('Batch from dataloader')



if  __name__=="__main__":
    train_data = ScanNet2d(root_data,'train',(968, 1296),20)

    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size,  sample['Y'].size)

    dataloader = DataLoader(train_data, batch_size=4, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())

        if i==1:
            plt.figure()
            showbatch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break



        
        


