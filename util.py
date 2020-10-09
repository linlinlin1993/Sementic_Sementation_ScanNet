from matplotlib.image import imread
import numpy as np
import os
import csv
from PIL import Image
import random
from PIL import Image

def map_label_image(image, label_mapping):
    #image = image.astype(np.uint8)
    mapped = np.copy(image)
    # print(np.amax(image))
    # print(type(image[0,0]))
    # h,w = image.shape
    #print(label_mapping[210])
    # for v in range(h):
    #     for u in range(w):
    #         if image[v,u]==1066:
    #             print(v,u)
    for k,v in label_mapping.items():
        # if k==42:
        #     print(v)
        #     print(np.sum(image==42))
        mapped[image==k] = v
    # print('mmms {}'.format(np.amax(mapped)))
    # print('mmmm {}'.format(np.amax(mapped.astype(np.uint8))))
    return mapped.astype(np.uint8)

def read_label_mapping(filename, label_from='id', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    #print(type(list(mapping.keys())[0]))
    # print(mapping.keys())
    # if 210 in list(mapping.keys()):
    #    print('iiiii')
    return mapping

# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def label_mapping(image, label_map_file, label_from = 'id', label_to = 'nyu40id'):
    label_map = read_label_mapping(label_map_file, label_from, label_to)
    # print('check label map')
    # print(label_map[42])
    # print(label_map[133])
    # print(label_map[218])
    mapped_image = map_label_image(image, label_map)

    return mapped_image

#def visulaize_output(outputs,labels,color_mapping,n_class):
   # outputs = outputs.data.cpu().numpy()
   # N, _, h, w = outputs.shape
   # preds = outputs.transpose(0,2,3,1).reshape(-1, n_class+1).argmax(axis=1).reshape(N,h,w)
   # targets = labels.cpu().numpy().reshape(N,h,w)
    #align color
    #preds_=[]
    #targets_=[]
    #for p,t in zip(preds,targets):
     #   p_v, t_v = alignColor(p,t,color_mapping)
      #  preds_.append(p_v)
       # targets_.append(t_v)
    #return np.array(preds_),np.array(targets_)

def visulaize_segmentation(output, label, img):
    h,w = output.shape
    pred = output.astype(np.uint8)
    target = label.astype(np.uint8)
    color_palette = create_color_palette_valid()
    pred_v = np.zeros([h,w,3], dtype=np.uint8)
    target_v = np.zeros([h,w,3], dtype=np.uint8)
    for idx, color in enumerate(color_palette):
        pred_v[pred==idx]=color
        target_v[target==idx]=color
    #img= img.data.cpu().numpy()
    print(img.shape)
    print(pred_v.shape)
    img= 255*img
    img = img.astype(np.uint8)
    Img = Image.fromarray(img,'RGB')
    pred_viz = Image.fromarray(pred_v, 'RGB')
    target_viz = Image.fromarray(target_v, 'RGB')

    images = [Img, pred_viz, target_viz]
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    

    return new_im



def visulaize_output(outputs,labels,n_class):
    outputs = outputs.data.cpu().numpy()
    N, _, h, w = outputs.shape
    preds_ = outputs.transpose(0,2,3,1).reshape(-1, n_class+1).argmax(axis=1).reshape(N,h,w)
    targets_ = labels.cpu().numpy().reshape(N,h,w)
    #align color
    preds = preds_.astype(np.uint8)
    targets = targets_.astype(np.uint8)
    color_palette = create_color_palette_valid()
    preds_v = np.zeros([N, h, w, 3], dtype=np.uint8)
    targets_v = np.zeros([N, h, w, 3], dtype=np.uint8)
    for p,t,p_v,t_v in zip(preds, targets, preds_v, targets_v):
        for idx, color in enumerate(color_palette):
            p_v[p==idx] = color
            t_v[t==idx] = color
    return preds_v, targets_v
def alignColor(pred,target,color_mapping):
    h,w = pred.shape
    print("shape")
    print(h,w)
    p_v = np.zeros((h,w,3))
    t_v = np.zeros((h,w,3))
    for v in range(h):
        for u in range(w):
            #print("p in pred {}".format(int(pred[v,u])))
            #rint("p in target {}".format(int(target[v,u])))
            p_v[v,u] = color_mapping[int(pred[v,u])]
            t_v[v,u] = color_mapping[int(target[v,u])]
    return p_v,t_v

def GenerateColorMapping(n_class):
    color_mapping ={}
    color_mapping[0]=[0,0,0]
    for i in range(1,n_class+1):
        color_mapping[i]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
    return color_mapping

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#valid_class = [2,1,6,5,3,39,8,7,40,10,16,38,24,18,4,9,12,14,25,32]
# 1	wall
# 2	floor
# 3	cabinet
# 4	bed
# 5	chair
# 6	sofa
# 7	table
# 8	door
# 9	window
# 10	bookshelf
# 11	picture
# 12	counter
# 13	blinds
# 14	desk
# 15	shelves
# 16	curtain
# 17	dresser
# 18	pillow
# 19	mirror
# 20	floor mat
# 21	clothes
# 22	ceiling
# 23	books
# 24	refridgerator
# 25	television
# 26	paper
# 27	towel
# 28	shower curtain
# 29	box
# 30	whiteboard
# 31	person
# 32	nightstand
# 33	toilet
# 34	sink
# 35	lamp
# 36	bathtub
# 37	bag
# 38	otherstructure
# 39	otherfurniture
# 40	otherprop
def create_color_palette_valid():
    valid_class = [1,3,2,8,39,6,16,24,40,10,12,9,38,5,7,4,18,14,25,32]
    color_palette = create_color_palette()
    color_palette_valid = [color_palette[i] for i in valid_class]
    color_palette_valid.append((112, 128, 144)) # align other classes to the color of sink
    return color_palette_valid
def create_color_palette():
    return [
       (0, 0, 0),         #[2,1,6,5,3,39,8,7,40,10,16,38,24,18,4,9,12,14,25,32]
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
def convert_to_valid_trainId(label, class_mapping):
    mapped=np.copy(label)
    for k,v in class_mapping.items():
        # if k==42:
        #     print(v)
        #     print(np.sum(image==42))
        mapped[label==k] = v
    # print('mmms {}'.format(np.amax(mapped)))
    # print('mmmm {}'.format(np.amax(mapped.astype(np.uint8))))
    return mapped.astype(np.uint8)
    
def valid_class_map():
    #valid_class = [1,2,3,6,39,8,16,38,5,10,7,24,40,9,4,18,12,14,25,32] 
    valid_class = [1,3,2,8,39,6,16,24,40,10,12,9,38,5,7,4,18,14,25,32]
    #[ 2  0  1  6  5  3 39  8  7 40 10 16 38 24 18  4  9 12 14 25 32 27 33 37 34 35 11 15 36 30 13 29 17 19 31 21 22 23 26 28 20]
    class_mapping = {}
    classes = np.arange(41)
    for index,cl in enumerate(classes):
        if cl in valid_class:
            class_mapping[cl]=valid_class.index(cl)
        else:
            if cl!=0:
                class_mapping[cl]=20
            else:
                class_mapping[cl]=255
    print(class_mapping)
    return class_mapping

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

##########
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    print("acc_cls: {}".format(acc_cls))
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc
#########
def main():
    label_map_file = "C:\\Users\\ji\\Documents\\ScanNet-master\\data\\scannetv2-labels.combined.tsv"
    label = Image.open("C:\\Users\\ji\\Documents\\ScanNet-master\\data\\scene0000_00\\scene0000_00_2d-label-filt\\label-filt\\20.png")
    label=np.array(label)
    mapped_image = label_mapping(label, label_map_file)

    print(min(min(x) for x in mapped_image))


if __name__=="__main__":
    main()