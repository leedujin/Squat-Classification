import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import random
import numpy as np
import glob
from PIL import Image
import json

class VideoDataset(Dataset):
    def __init__(self, file_path):

        with open(file_path, 'r') as json_file:
            file = json.load(json_file)
            
        self.data_path_list = []
        self.label_list = []

        for i in file.keys():
            folder_path = file[i]["folder_path"].split("/")
            folder_path[0] = "VideoPose3D"
            del folder_path[1:4]
            file_list = [j for j in os.listdir("VideoPose3D") if j.endswith("_3d")]
            for k in file_list:
                folder_path[2] = folder_path[2].split(".")[0] + ".mp4_3d_skeleton.npy"
                if folder_path[2] in os.listdir("VideoPose3D/" + k):
                    folder_path[1] = k
                    break

            if folder_path[1] in file_list:
                file[i]["folder_path"] = "/".join(folder_path)
                self.data_path_list.append(file[i]["folder_path"])
                self.label_list.append(file[i]["label"])

    def __getitem__(self, index):
        data_path = self.data_path_list[index]
        data = torch.FloatTensor(np.load(data_path))[-299:]
        
        
        if self.label_list is not None:
            label = int(self.label_list[index])
            return data, label
        else:
            return data
        
    def __len__(self):
        return len(self.data_path_list)


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('Squat/Video_Dataset/Divided/train.json'),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True)
    
    dataloaders['val'] = torch.utils.data.DataLoader(VideoDataset('Squat/Video_Dataset/Divided/val.json'),
                                                      batch_size=args.val_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('Squat/Video_Dataset/Divided/test.json'),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True)
    return dataloaders