# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
from ncf import NeuMF
#from utils import create_ml_1m_dataset
from preprocess import dataprocess
def run():
    ###################################### prepare data ###########################################
    #_, test = create_ml_1m_dataset('./ratings100.dat')
    _, __, ___, user_test, item_test, neg_item_test, = dataprocess()
    ################################### init & load model #########################################
    net = NeuMF()
    net.load_state_dict(torch.load('ncf.pth'))
    ######################################## evaluate #############################################
    K = 10
    # user_test = torch.from_numpy(test[0])
    # item_test = torch.from_numpy(test[1])
    # neg_item_test = torch.from_numpy(test[2])
    net.eval()
    pos_pred = - net.eval_pred(user_test, item_test)
    pos_pred = pos_pred.unsqueeze(1)
    for i in range(101):
        if i == 0:
            j = neg_item_test[i]
            neg_pred = - net.eval_pred(user_test, j)
            neg_pred = neg_pred.unsqueeze(1)
            neg_pred_lst = torch.cat([pos_pred, neg_pred], dim=-1)
        else:
            j = neg_item_test[i]
            neg_pred = - net.eval_pred(user_test, j)
            neg_pred = neg_pred.unsqueeze(1)
            neg_pred_lst = torch.cat([neg_pred_lst, neg_pred], dim=-1)
    # neg_pred = - net.eval_pred(user_test, neg_item_test)
    pred = neg_pred_lst

    rank0 = pred.argsort()
    rank1 = rank0.argsort()
    rank2 = rank1[:, 0]
    rank = rank2

    # rank = pred.argsort().argsort()[:, 0]
    hr, ndcg, mrr = 0.0, 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1.0
            ndcg += 1.0 / np.log2(r + 2)
            if r <= 1:
                mrr += 2.0
            else:
                mrr += 1.0 / r
    print("HR : {}, NDCG : {}, MRR : {}".format(hr / len(rank), ndcg / len(rank), mrr / len(rank)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(66 * '=')

if __name__ == "__main__":
    run()