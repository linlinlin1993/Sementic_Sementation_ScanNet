import os
import torch
#import LoadData
from torch.utils.data import DataLoader
from matplotlib.image import imread
#from LoadData import ScanNet2d
from Dataset_processing import ComputeMeanofInput
import numpy as np
import imageio
import util
import pandas as pd
from matplotlib.image import imread
import imageio
from util import GenerateColorMapping,create_color_palette_valid
from PIL import Image
from LoadData import ScanNet2d
import matplotlib.pyplot as plt
def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    imageio.imwrite(filename, vis_image)

def alignColor(pred,color_mapping):
    h,w = pred.shape
    print("shape")
    print(h,w)
    p_v = np.zeros((h,w,3))
    for v in range(h):
        for u in range(w):
            #print("p in pred {}".format(int(pred[v,u])))
            #rint("p in target {}".format(int(target[v,u])))
            p_v[v,u] = color_mapping[int(pred[v,u])]
            #print(p_v[v,u])
    return p_v
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]

def analysis_input(train_file):
    classes=np.zeros((41))
    train_data = ScanNet2d(csv_file=train_file, phase = 'train', MeanRGB=np.array([0.0,0.0,0.0]),n_class=40,trainsize=(484, 648),data_analysis=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
    for i,batch in enumerate(train_loader):
        target = batch['l'].cpu().numpy().reshape(1, 484*648)
    
        for cl in range(41):
            target_cls = target==cl
        
            classes[cl] +=target_cls.sum()

    # plt.plot(x,classes,".")
    # plt.show()
    # plt.savefig("input_analysis.png")
    return classes
def visualize_input_data(train_file):
    #MeanRGB = np.load(".\\Mean_train_1000from012.npy")
    MeanRGB = np.array([0,0,0])
    train_data = ScanNet2d(csv_file=train_file, phase = 'train', trainsize=(484, 648), MeanRGB=MeanRGB, data_analysis=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    for i,batch in enumerate(train_loader):
        label = batch['l'].cpu().numpy().reshape(1,484,648)[0]
        
        inp = batch['X'].cpu().numpy().transpose(0,2,3,1)[0]
        print(inp.shape)
        vis_image = np.zeros([484, 648, 3], dtype=np.uint8)
        color_palette = create_color_palette()
        for idx, color in enumerate(color_palette):
            vis_image[label==idx] = color
        imageio.imwrite(".\\data_\\"+str(i)+"_label.png", vis_image)
        imageio.imwrite(".\\data_\\"+str(i)+"_render.png", inp)




def computer_class_weights(train_file):
    classses_ = analysis_input(train_file)
    sortclass = classses_.argsort()[::-1]
    classses_[::-1].sort()
    valid = classses_[:21]
    valid_class = sortclass[:21]
    not_valid = classses_[21:]
    nvalid_class = sortclass[21:]
    not_valid_f = np.sum(not_valid)+valid[1]
    valid = np.delete(valid, 1)
    valid_class= np.delete(valid_class,1)
    print(valid.shape)
    all_cf = np.concatenate([valid,np.array([not_valid_f])])
    import statistics
    median_f = statistics.median(all_cf) 
    class_weight = median_f/all_cf
    print(median_f)
    return class_weight
if __name__=='__main__':
    # root_dir=".\\"
    # train_file = os.path.join(root_dir, "train_1000from012.csv")
    # label_map_file="C:\\Users\\ji\\Documents\\ScanNet-master\\data\\scannetv2-labels.combined.tsv"
    # classses_ = analysis_input(train_file)
    # sortclass = classses_.argsort()[::-1]
    # print("sortclass: {}".format(sortclass))
    # classses_[::-1].sort()
    
    # ## remove unlabel samples
    # classses = np.delete(classses_, list(sortclass).index(0))
    # sortclass_without0 = np.delete(sortclass, list(sortclass).index(0))
    # valid = classses[:20]
    # valid_class = sortclass_without0[:20]
    # not_valid = classses[20:]
    # nvalid_class = sortclass_without0[20:]
    # not_valid_f = np.sum(not_valid)
    # print(valid.shape)
    # all_cf = np.concatenate([valid,np.array([not_valid_f])])
    # import statistics
    # median_f = statistics.median(all_cf)
    # class_weight = median_f/all_cf
    # np.save("class_weight_1000from012_half_nounlabel.npy",class_weight)
    # print(valid_class)
    # print(classses_)
    # print(all_cf)
    # print(median_f)
    # print(median_f/all_cf)
    

     


    visualize_input_data(".\\train_300from789.csv")
    # train_file = os.path.join(root_dir, "train_1000from012.csv")
    # train_data = ScanNet2d(csv_file=train_file, phase = 'train', MeanRGB=np.array([0.0,0.0,0.0]),n_class=20,trainsize=(484, 648),data_analysis=False)
    # train_loader = DataLoader(train_data, batch_size=2, shuffle=False, num_workers=1)
    # for i,batch in enumerate(train_loader):
    #     target = batch['l']
    #     if i ==0:
    #         print("target")
    #         print(target)
    #         t = target.view(2,484,648,1).repeat(1,1,1,21)
    #         print(t)
    #         print(t.shape)
    #         print(t>=0)
    