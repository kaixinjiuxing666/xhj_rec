# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
from deepfm import deepFM
#from utils import create_ml_1m_dataset
from preprocess import dataprocess

def run():
    ###################################### prepare data ###########################################
    _, __, ___, user_test, item_test, neg_item_test, = dataprocess()
    ################################### init & load model #########################################
    net = deepFM()
    net.load_state_dict(torch.load('deepfm.pth'))
    ######################################## recall #############################################
    one_user = user_test[0].view(1,3).repeat(item_test.shape[0],1)
    net.eval()
    pred = - net.eval_pred(one_user, item_test)
    rank = pred.argsort().tolist()

    print(rank[:50])
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    return rank[:10]

if __name__ == "__main__":
    run()