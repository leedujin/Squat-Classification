import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import sys
from scipy import stats
from tqdm import tqdm
import argparse
import numpy as np
import glob
import logging
from models import CNN
from loader import VideoDataset, get_dataloaders
from config import get_parser
from utils import get_logger,log_and_print, compute_acc

sys.path.append('../')
torch.backends.cudnn.enabled = True
feature_dim = 1024


if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    base_logger = get_logger(f'exp/CNN.log', args.log_info)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cnn = CNN(3).cuda()
    dataloaders = get_dataloaders(args)

    optimizer = torch.optim.Adam([*cnn.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    acc_best = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):

        log_and_print(base_logger, f'Epoch: {epoch}')

        for split in ['train', 'val', 'test']:
            true_labels = []
            pred_labels = []

            if split == 'train':
                cnn.train()
                torch.set_grad_enabled(True)
            else:
                cnn.eval()
                torch.set_grad_enabled(False)

            for data, labels in tqdm(dataloaders[split]):
                data = data.cuda()
                true_labels.extend(labels)

                probs = cnn(data)

                if split == 'train':
                    #loss = loss_function_v2(sigma, data['final_score'].float().cuda(), mu)
                    loss = criterion(probs, labels.cuda())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                pred_labels.extend(probs.argmax(dim=1).tolist())

            acc = compute_acc(true_labels, pred_labels)

            log_and_print(base_logger, f'{split} accuracy: {acc}')

        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            log_and_print(base_logger, '##### New best Accuracy #####')
            path = 'ckpts/' + str(acc) + '.pt'
            torch.save({'epoch': epoch,
                            'cnn': cnn.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'acc_best': acc_best}, path)