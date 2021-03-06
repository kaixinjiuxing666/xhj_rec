# -*- coding: utf-8 -*-
import numpy as np
from torch.utils import data
import torch
import time
from convncf import ConvNCF
from torch import nn
from utils import create_ml_1m_dataset

def run():
    ###################################### prepare data ###########################################
    # user = torch.tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],])
    # item = torch.tensor([[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],])
    # neg_item = torch.tensor([[9],[9],[9],[9],[9],[9],[9],[9],[9],[9],])
    # user = torch.tensor([[1,1,1,1,1,1,1,1,1,1]])
    # item = torch.tensor([[5,5,5,5,5,5,5,5,5,5]])
    # neg_item = torch.tensor([[9,9,9,9,9,9,9,9,9,9,]])
    train, test = create_ml_1m_dataset('./ratings100.dat')
    user = torch.from_numpy(train[0])
    item = torch.from_numpy(train[1])
    neg_item = torch.from_numpy(train[2])
    ############################################# load data #######################################
    def load_array(data_arrays, batch_size, is_train=True):  # @save
        """构造一个PyTorch数据迭代器。"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    batch_size = 256
    epoch_num = 10
    data_iter = load_array((user, item, neg_item), batch_size)
    #print(next(iter(data_iter)))
    ################################################## init model ##################################
    net = ConvNCF()

    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    ################################################## train model ##################################
    for epoch in range(epoch_num):
        total_loss = 0
        epoch_time = time.time()
        for user,pos_item,neg_item in data_iter:
            optimizer.zero_grad()
            losses = net.calculate_loss(user, pos_item, neg_item)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'epoch {epoch + 1}, loss {total_loss:f}, 1 epoch time {time.time() - epoch_time:.2f}')
    ############################################## evaluate ######################################
        torch.save(net.state_dict(), "convncf.pth")
        K = 10
        user_test = torch.from_numpy(test[0])
        item_test = torch.from_numpy(test[1])
        neg_item_test = torch.from_numpy(test[2])
        net.eval()
        pos_pred = - net.eval_pred(user_test, item_test)
        for i in range(101):
            if i == 0:
                j = neg_item_test[:,i]
                neg_pred = - net.eval_pred(user_test, j)
                neg_pred_lst = torch.cat([pos_pred, neg_pred], dim=-1)
            else:
                j = neg_item_test[:,i]
                neg_pred = - net.eval_pred(user_test, j)
                neg_pred_lst = torch.cat([neg_pred_lst, neg_pred], dim=-1)
        #neg_pred = - net.eval_pred(user_test, neg_item_test)
        pred = neg_pred_lst

        # rank0 = pred.argsort()
        # rank1 = rank0.argsort()
        # rank2 = rank1[:, 0]
        # rank = rank2

        rank = pred.argsort().argsort()[:, 0]
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
        print(55*'=')
        #return hr / len(rank), ndcg / len(rank)

if __name__ == "__main__":
    run()