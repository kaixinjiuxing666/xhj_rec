# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.init import normal_
from recbole.model.layers import MLPLayers



class NeuMF(nn.Module):
    def __init__(self):
        super().__init__()

        # load dataset info

        # load parameters info

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(9999, 64)
        self.item_mf_embedding = nn.Embedding(9999, 64)
        self.user_mlp_embedding = nn.Embedding(9999, 64)
        self.item_mlp_embedding = nn.Embedding(9999, 64)


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

        # parameters initialization
        # if self.use_pretrain:
        #     self.load_pretrain()
        # else:
        #     self.apply(self._init_weights)

    # def load_pretrain(self):
    #     r"""A simple implementation of loading pretrained parameters.
    #
    #     """
    #     mf = torch.load(self.mf_pretrain_path)
    #     mlp = torch.load(self.mlp_pretrain_path)
    #     self.user_mf_embedding.weight.data.copy_(mf.user_mf_embedding.weight)
    #     self.item_mf_embedding.weight.data.copy_(mf.item_mf_embedding.weight)
    #     self.user_mlp_embedding.weight.data.copy_(mlp.user_mlp_embedding.weight)
    #     self.item_mlp_embedding.weight.data.copy_(mlp.item_mlp_embedding.weight)
    #
    #     for (m1, m2) in zip(self.mlp_layers.mlp_layers, mlp.mlp_layers.mlp_layers):
    #         if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
    #             m1.weight.data.copy_(m2.weight)
    #             m1.bias.data.copy_(m2.bias)
    #
    #     predict_weight = torch.cat([mf.predict_layer.weight, mlp.predict_layer.weight], dim=1)
    #     predict_bias = mf.predict_layer.bias + mlp.predict_layer.bias
    #
    #     self.predict_layer.weight.data.copy_(0.5 * predict_weight)
    #     self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))

        return output.squeeze()

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
