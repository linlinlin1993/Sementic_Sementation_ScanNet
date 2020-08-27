from matplotlib.image import imread
import numpy as np
import os
import csv


def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k,v in label_mapping.iteritems():
        mapped[image==k] = v
    return mapped.astype(np.unit8)

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(mapping.keys()[0]):
        mapping = {int(k):v for k,v in mapping.items()}
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
    mapped_image = map_label_image(image, label_map)

    return mapped_image