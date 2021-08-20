# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class ConvNCFBPRLoss(nn.Module):

    def __init__(self):
        super(ConvNCFBPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        distance = pos_score - neg_score
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        return loss

class ConvNCF(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers and loss
        self.user_embedding = nn.Embedding(9999, 64)
        self.item_embedding = nn.Embedding(9999, 64)
        self.cnn_layers = nn.Sequential(
                      nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(4, 4)),
                      nn.ReLU(),
                      nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4)),
                      nn.ReLU(),
                      nn. Conv2d(32, 32, kernel_size=(2, 2), stride=(2, 2)),
                      nn.ReLU(),
                      nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(2, 2)),
                      nn.ReLU(),
        )
        self.predict_layers = nn.Sequential(
                      nn.Dropout(p=0.2, inplace=False),
                      nn.Linear(in_features=32, out_features=1, bias=True),
        )
        self.loss = ConvNCFBPRLoss()
    def forward(self, user, item):  # (2048,) batch=2048
        ################
        user = user.squeeze()
        item = item.squeeze()

        ################
        user_e = self.user_embedding(user)  # (2048,64)
        item_e = self.item_embedding(item)  # (2048,64)

        interaction_map = torch.bmm(user_e.unsqueeze(2), item_e.unsqueeze(1))  # (2048,64,64)
        interaction_map = interaction_map.unsqueeze(1)  # (2048,1,64,64)

        cnn_output = self.cnn_layers(interaction_map)  # (2048,32,1,1)
        cnn_output = cnn_output.sum(axis=(2, 3))  # (2048,32)

        prediction = self.predict_layers(cnn_output)  # (2048,1)
        prediction = prediction.squeeze()  # (2048,)

        return prediction


    # def forward(self, user, item):#(2048,) batch=2048
    #     user_e = self.user_embedding(user)#(2048,64)
    #     item_e = self.item_embedding(item)#(2048,64)
    #
    #     user_e = user_e.transpose(1,2).contiguous()# 维度变换
    #     user_e = user_e.repeat(1,1,2)# 维度重复
    #
    #     interaction_map = torch.bmm(user_e, item_e)#(2048,64,64)
    #     interaction_map = interaction_map.unsqueeze(1)#(2048,1,64,64)
    #
    #     cnn_output = self.cnn_layers(interaction_map)#(2048,32,1,1)
    #     cnn_output = cnn_output.sum(axis=(2, 3))#(2048,32)
    #
    #     prediction = self.predict_layers(cnn_output)#(2048,1)
    #     prediction = prediction.squeeze()#(2048,)
    #
    #     return prediction


    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = [0.1,0.1]
        loss_1 = reg_1 * self.user_embedding.weight.norm(2)
        loss_2 = reg_1 * self.item_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.cnn_layers.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        for name, parm in self.predict_layers.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def calculate_loss(self, user, pos_item, neg_item):

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)

        loss = self.loss(pos_item_score, neg_item_score)
        opt_loss = loss + self.reg_loss()

        #return loss
        return opt_loss

    def eval_pred(self, user, item):
        pred = self.forward(user, item)
        pred = pred.unsqueeze(1)
        return pred


def run():
    user = torch.tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],])
    item = torch.tensor([[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],])
    neg_item = torch.tensor([[9], [9], [9], [9], [9], [9], [9], [9], [9], [9], ])
    net = ConvNCF()
    pos_pred = - net.eval_pred(user, item)
    neg_pred = - net.eval_pred(user, neg_item)
    output = torch.cat([pos_pred,neg_pred],dim=-1)
    print(pos_pred)
    print(neg_pred)
    print(output)
    print(output.shape)

#run()


