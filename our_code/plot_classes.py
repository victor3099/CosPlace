from matplotlib import pyplot as plt
import torch
import sys
import numpy as np
from glob import glob
def plot_labels(filename):
    classes_per_group, images_per_class = torch.load(filename)
    #print(classes_per_group)
    print(f'Number of classes: {len(images_per_class)}')
    y = []
    for k in images_per_class:    
        if (k[2] != 0):
            print()
        number_of_classes = len(images_per_class[k])       
        y.append(number_of_classes)
    plt.bar(np.arange(len(y)),y)
    plt.show()

def check_headers():
    dataset_folder = "../small/test"
    images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
    images_metadatas = [p.split("@") for p in images_paths]
    header_set = set()
    for m in images_metadatas:
        header = m[9]
        header_set.add(header) 
    print(header_set)

    images_metadatas = [p.split("@") for p in images_paths]

if __name__ == "__main__":
    #plot_labels(sys.argv[1])
    check_headers()