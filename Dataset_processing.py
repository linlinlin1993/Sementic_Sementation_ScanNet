#import pandas
import os,sys,argparse
import csv
import numpy as np
from matplotlib.pyplot import imread
import pandas as pd
import PIL.Image
import random
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',default="C:\\Users\\ji\\Documents\\ScanNet-master\\data", help="path of data")
parser.add_argument('--phase', default = "val")
parser.add_argument('--output_file', default="C:\\Users\\ji\\Documents\\FCN-VGG16\\val_300from0123.csv")
parser.add_argument('--debug', default=True)
opt = parser.parse_args()

# if not opt.output_file:
#     opt.output_file = os.path.join(opt.data_path,opt.phase+'.csv')
#     print(opt.output_file)

def writeDataInfoToCSV(num_train):
    scenes = os.listdir(opt.data_path)
    scenes = [os.path.join(opt.data_path,s) for s in scenes]
    print(scenes)

    with open(opt.output_file, mode ='w') as csv_file:
        headnames = ['sceneId', 'inputfile', 'labelfile']
        mywriter = csv.writer(csv_file)
        mywriter.writerow(headnames)
        #scenes.sort(key=lambda x:int(x.split('e')[-1]) 
        train_map = secene_train_mappin()
     
        ziplist = []
        for s in scenes[:5]:
            if (os.path.isdir(s)):
                sceneId = s.split('e')[-1]
                
                if opt.debug:
                    print(sceneId)
                files = os.listdir(s)
                files =[os.path.join(s,f) for f in files]
                #print(files)
                if opt.phase == 'train':
                    print("listing train data for "+sceneId +"...")
                    for f in files:
                        if os.path.isdir(f) and '2d-rendered-RGB' in f:
                            print("listing the rendered color images...")
                            imgs_ = os.listdir(f)
                            imgs_.sort(key=lambda x:int(os.path.splitext(x)[0]))
                            imgs=[]
                            [imgs.append(os.path.join(f,img)) for img in imgs_]
                            #print(imgs[10])
                        if os.path.isdir(f) and 'label' in f and '2d' not in f:
                            print("listing the label...")
                            print(f)
                            labels_ = os.listdir(f)
                            labels_.sort(key=lambda x:int(os.path.splitext(x)[0]))
                            labels=[]
                            [labels.append(os.path.join(f,label)) for label in labels_]
                    #index = random.sample(range(len(imgs)),300)
                    #if sceneId in ["0000_00"]
                    for i in range(len(imgs)):
                        i=int(i)
                        frame = int(imgs[i].split("\\")[-1].split('.')[0])
                        if opt.debug:
                            print([sceneId,imgs[i],labels[frame]])
                        ziplist.append([sceneId, imgs[i], labels[frame]])
                       # mywriter.writerow([sceneId,imgs[i],labels[frame]])    
                   # for count,im in enumerate(imgs):
                    #    frame = int(im.split("\\")[-1].split('.')[0])
                     #   if opt.debug:
                      #      print([sceneId,im,labels[frame]])
                       # if frame == 0:
                       # mywriter.writerow([sceneId,im,labels[frame]])    
                  
                if opt.phase == 'val':
                    print("listing val data for"+sceneId + "...")
                    for f in files:
                        if os.path.isdir(f) and '2d-rendered-RGB' in f:
                            print("listing the rendered color images...")
                            imgs_ = os.listdir(f)
                            imgs_.sort(key=lambda x:int(os.path.splitext(x)[0]))
                            imgs=[]
                            [imgs.append(os.path.join(f,img)) for img in imgs_]
                            #print(imgs[10])
                        if os.path.isdir(f) and 'label' in f and '2d' not in f:
                            print("listing the label...")
                            print(f)
                            labels_ = os.listdir(f)
                            labels_.sort(key=lambda x:int(os.path.splitext(x)[0]))
                            labels=[]
                            [labels.append(os.path.join(f,label)) for label in labels_]
                    train_frames = train_map[sceneId]
                    val_frames = [i for i in list(range(len(imgs))) if int(imgs[i].split("\\")[-1].split('.')[0]) not in train_frames]
                    #index = random.sample(val_frames,30)
                    for n in val_frames:
                        n= int(n)
                        frame = int(imgs[n].split("\\")[-1].split('.')[0])
                        if opt.debug:
                            print([sceneId,imgs[n],labels[frame]])
                        ziplist.append([sceneId, imgs[n], labels[frame]])
                        #mywriter.writerow([sceneId,imgs[n],labels[frame]])  
                if opt.phase == 'test':
                    print("listing test data for " + sceneId + "...")
                    for f in files:
                        if os.path.isdir(f) and '2d-origiRGB' in f:
                            print("listing original color images...")
                            imgs_ = os.listdir(f)
                            imgs_.sort(key=lambda x:int(os.path.splitext(x)[0]))
                            imgs=[]
                            [imgs.append(os.path.join(f,img)) for img in imgs_]
                        if os.path.isdir(f) and '2d-label-filt' in f:
                            print("listing labels...")
                            labels_ = os.listdir(os.path.join(f,"label-filt"))
                            labels_.sort(key=lambda x:int(os.path.splitext(x)[0]))
                            labels=[]
                            [labels.append(os.path.join(f,"label-filt",label)) for label in labels_]
                   #if os.path.isfile(f) and os.path.splitext(f)[1]=='.png':
                    index = random.sample(range(len(imgs)),100)
                    #for count,im in enumerate(imgs):
                    for i in index:
                        frame = int(imgs[i].split("\\")[-1].split('.')[0])
                        if opt.debug:
                            print([sceneId,imgs[i],labels[frame]])
                        mywriter.writerow([sceneId,imgs[i],labels[frame]])
        index = random.sample(list(range(len(ziplist))),num_train)
        train_data = [ziplist[i] for i in index]
        mywriter.writerows(train_data)
def secene_train_mappin(corr_train_file = "C:\\Users\\ji\\Documents\\FCN-VGG16\\train_3000from0123.csv"):
    
    train_data = pd.read_csv(corr_train_file)
    input_frame_mapping ={"0000_00": [], "0001_00":[], "0002_00": [], "0003_00": []}
    #se_id = ["0000_00","0001_00","0002_00"]
    for idx in range(len(train_data)):

        se = train_data.iloc[idx,0]
        #if str(sceneId)==se:
        im_file=train_data.iloc[idx,1]
        frame = int(im_file.split("\\")[-1].split('.')[0])
        input_frame_mapping[str(se)].append(frame)
    print(input_frame_mapping)
    return input_frame_mapping

def ComputeMeanofInput(csv_file):
    data = pd.read_csv(csv_file)
    print("file name: {}",format(data.iloc[0,1]))
    
    img0 = imread(data.iloc[0,1])[:,:,:3]
    H,W,_ = img0.shape
    img0=np.zeros((H,W,3))
    print(W) 
    for i in range(len(data)):
        img_name= data.iloc[i,1]
        img= imread(img_name)[:,:,:3]
        img0= img0+img
    PixelSum=np.array([0.0,0.0,0.0])
    for v in range(H):
        for u in range(W):
            PixelSum += img0[v,u]
    Mean = PixelSum/(len(data)*W*H)
    print("Mean")
    print(Mean)
    return Mean

def main():
   # writeDataInfoToCSV(300)
    #secene_train_mappin()
    # if not opt.o
    # utput_file:
    #     opt.output_file = os.path.join(opt.data_path,opt.phase+'.csv')
    #     print(opt.output_file)

    #     print(opt.output_file)
    #     writeDataInfoToCSV()
    train_file = ".\\train_3000from0123.csv"
    Mean = ComputeMeanofInput(train_file)
    train_f= train_file.split("\\")[-1].split(".")[0]
    np.save("Mean_{}{}".format(train_f,".npy"),Mean)
                    
        
if __name__=='__main__':
    main()

