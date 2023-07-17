import torch
import logging
import math
import torch.nn as nn
import numpy as np

alpha, beta = torch.Tensor([0.6]).cuda(), torch.Tensor([0.4]).cuda()


def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("-" * 30 + log_info + "-" * 30)
    return logger

def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def compute_acc(true_list, pred_list):
    point = 0
    for i in range(len(true_list)):
        if true_list[i] == pred_list[i]:
            point += 1
    
    acc = point/len(true_list)

    return acc