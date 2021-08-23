# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.init import normal_

class NeuMF(nn.Module):
    def __init__(self):
        super().__init__()

        # load dataset info

        # load parameters info

        # define layers and loss
        # self.user_mf_embedding = nn.Embedding(9999, 64)
        # self.item_mf_embedding = nn.Embedding(9999, 64)
        # self.user_mlp_embedding = nn.Embedding(9999, 64)
        # self.item_mlp_embedding = nn.Embedding(9999, 64)
        self.user_mf_embedding = nn.Linear(3, 64)
        self.item_mf_embedding = nn.Linear(9, 64)
        self.user_mlp_embedding = nn.Linear(3, 64)
        self.item_mlp_embedding = nn.Linear(9, 64)

        self.mlp_layers = nn.Sequential(nn.Dropout(p=0.1, inplace=False),
                                        nn.Linear(in_features=128, out_features=128, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1, inplace=False),
                                        nn.Linear(in_features=128, out_features=64, bias=True),
                                        nn.ReLU(),
                                        )
        self.predict_layer = nn.Linear(in_features=128, out_features=1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)


    # def forward(self, user, item):
    #     user_mf_e = self.user_mf_embedding(user)
    #     item_mf_e = self.item_mf_embedding(item)
    #     user_mlp_e = self.user_mlp_embedding(user)
    #     item_mlp_e = self.item_mlp_embedding(item)
    #
    #     mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
    #     mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
    #     output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
    #
    #     return output.squeeze()


    def forward(self, user, item):
        # user_e = user_e.transpose(1,2).contiguous()# 维度变换
        # user_e = user_e.repeat(1,1,2)# 维度重复

        user_mf_e = self.user_mf_embedding(user) # (batchsize, attribute num, embedding dim)
        item_mf_e = self.item_mf_embedding(item) # (batchsize, attribute num, embedding dim)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        #user_mf_e = user_mf_e.repeat(1, 3, 1)  # 维度重复
        mf_output = torch.mul(user_mf_e, item_mf_e)

        #user_mlp_e = user_mlp_e.repeat(1, 3, 1)  # 维度重复
        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))
        output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        output = output.squeeze()
        #output = torch.sum(output,dim=1)
        #output = torch.div(output, 9)
        return output

    def calculate_loss(self, user,pos_item,neg_item):
        pos_output = self.forward(user, pos_item)
        neg_output = self.forward(user, neg_item)
        output = torch.cat([pos_output,neg_output],dim=-1)

        x = torch.ones(pos_output.shape)
        y = torch.zeros(neg_output.shape)
        label = torch.cat([x,y],dim=-1)

        loss = self.loss(output,label)
        return loss

    def eval_pred(self, user, item):
        return self.forward(user, item)

    def dump_parameters(self):
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)


def run():
    # user = torch.tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],])
    # item = torch.tensor([[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],])
    # neg_item = torch.tensor([[9], [9], [9], [9], [9], [9], [9], [9], [9], [9], ])
    user = torch.tensor([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],])
    item = torch.tensor([[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],[5,6,7,8],])
    neg_item = torch.tensor([[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],[9,9,9,9],])
    net = NeuMF()
    pos_pred = - net.eval_pred(user, item)
    neg_pred = - net.eval_pred(user, neg_item)
    output = torch.cat([pos_pred,neg_pred],dim=-1)
    print(pos_pred)
    print(neg_pred)
    print(output)
    print(output.shape)

# run()