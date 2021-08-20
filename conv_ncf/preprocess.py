import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
import joblib
from word2vec import v2v
from word2vec_train import v2v_train


def dataprocess():

    file = './data/train_data_header_test.csv'
    print('---> {} <--- reading succeeded.'.format(file))
    df = pd.read_csv(file,encoding='gbk')
    ######################## user ###################################################
    user_lst = ['CostFrom','CostTo','Room','Sex',]
    user_ = []
    user_dic = {}

    # 填充空数据
    for user in user_lst[:-1]:
        try:
            x = round(df[user].median(), 2)
            df[user].fillna(x, inplace=True)
        except Exception as e:
            print(e)
            raise Exception('数据格式错误，请检查并修改！！！')

    df['Sex'].fillna('男', inplace=True)
    # 更改错误数据
    for x in df.index:
        try:
            if df.loc[x, 'CostFrom'] < 1000:
                df.loc[x, 'CostFrom'] = 1000
            if df.loc[x, 'CostFrom'] > 20000:
                df.loc[x, 'CostFrom'] = 20000
            if df.loc[x, 'CostTo'] < 1100:
                df.loc[x, 'CostTo'] = 1100
            if df.loc[x, 'CostTo'] > 20000:
                df.loc[x, 'CostTo'] = 21000
            if df.loc[x, 'Room'] < 2:
                df.loc[x, 'Room'] = 3
            if df.loc[x, 'Room'] > 5:
                df.loc[x, 'Room'] = 5
        except TypeError:
            df.drop(x, inplace=True)
            continue

    # 转换为浮点型列表
    for user in user_lst[:]:
        user_dic[user] = df[user].tolist()

    ######################## item #################################################
    item_lst = ['BuildingSize','FloorNumber','RoomCount','HallCount','ToiletCount',
                'Price','TotalPrice','EstateName','City_Name','Sq_Name']
    item_ = []
    item_dic = {}

    for item in item_lst[:7]:
        try:
            x = round(df[item].median(), 2)
            df[item].fillna(x, inplace=True)
        except Exception as e:
            print(e)
            raise Exception('数据格式错误，请检查并修改！！！')

    for item in item_lst[7:]:
        try:
            df[item].fillna('无', inplace=True)
        except Exception as e:
            print(e)
            raise Exception('数据格式错误，请检查并修改！！！')

    for x in df.index:
        if df.loc[x, 'BuildingSize'] < 20:
            df.loc[x, 'BuildingSize'] = 20
        if df.loc[x, 'BuildingSize'] > 300:
            df.loc[x, 'BuildingSize'] = 300

        if df.loc[x, 'FloorNumber'] <= 0:
            df.loc[x, 'FloorNumber'] = 0
        if df.loc[x, 'FloorNumber'] > 100:
            df.loc[x, 'FloorNumber'] = 100

        if df.loc[x, 'RoomCount'] <= 0:
            df.loc[x, 'RoomCount'] = 1
        if df.loc[x, 'RoomCount'] > 5:
            df.loc[x, 'RoomCount'] = 5

        if df.loc[x, 'HallCount'] <= 0:
            df.loc[x, 'HallCount'] = 1
        if df.loc[x, 'HallCount'] > 5:
            df.loc[x, 'HallCount'] = 2

        if df.loc[x, 'ToiletCount'] <= 0:
            df.loc[x, 'ToiletCount'] = 1
        if df.loc[x, 'ToiletCount'] > 3:
            df.loc[x, 'ToiletCount'] = 2

        if df.loc[x, 'Price'] < 2000:
            df.loc[x, 'Price'] = 2000
        if df.loc[x, 'Price'] > 30000:
            df.loc[x, 'Price'] = 30000

        if df.loc[x, 'TotalPrice'] < 10:
            df.loc[x, 'TotalPrice'] = 10
        if df.loc[x, 'TotalPrice'] > 300:
            df.loc[x, 'TotalPrice'] = 300

    for item in item_lst[:7]:
        item_dic[item] = df[item].tolist()



    ch_all_lst = df['Sex'].tolist() + df['EstateName'].tolist() +\
                 df['City_Name'].tolist() + df['Sq_Name'].tolist()
    v2v_train(ch_all_lst)


    user_dic['Sex'] = v2v(df['Sex'].tolist())
    item_dic['EstateName'] = v2v(df['EstateName'].tolist())
    item_dic['City_Name'] = v2v(df['City_Name'].tolist())
    item_dic['Sq_Name'] = v2v(df['Sq_Name'].tolist())


    #df['Price'].to_csv('new_data.csv')
    #################### user concat ###################
    # 按列拼接，每一列代表一个用户的各个属性
    user_ = np.concatenate((np.array([user_dic['CostFrom']]).T,
                        np.array([user_dic['CostTo']]).T,
                        np.array([user_dic['Room']]).T,
                        np.array([user_dic['Sex']]).T,
                        ), axis=1)
    # 使用 sklearn 使得初始数据标准化
    user_ = user_.astype(np.float32)
    user_ = np.array(user_)
    user_scaler = preprocessing.StandardScaler().fit(user_)
    user_ = user_scaler.transform(user_)
    #user_origin = user_scaler.inverse_transform(user_) # 将标准化的数据还原
    # 标准化后转换为 torch 格式的张量
    user_ = torch.tensor(user_)
    # d.resize(1,3)
    # print(user_.shape)
    #print(user_[100:105])

    ##################### item concat ###################

    item_ = np.concatenate((np.array([item_dic['BuildingSize']]).T,
                        np.array([item_dic['FloorNumber']]).T,
                        np.array([item_dic['RoomCount']]).T,
                        np.array([item_dic['HallCount']]).T,
                        np.array([item_dic['ToiletCount']]).T,
                        np.array([item_dic['Price']]).T,
                        np.array([item_dic['TotalPrice']]).T,
                        np.array([item_dic['EstateName']]).T,
                        np.array([item_dic['City_Name']]).T,
                        np.array([item_dic['Sq_Name']]).T,
                        ), axis=1)
    #print(item_[105:125])
    item_ = item_.astype(np.float32)
    # print(item_[100:105])
    item_ = np.array(item_)
    item_scaler = preprocessing.StandardScaler().fit(item_)
    item_ = item_scaler.transform(item_)
    #print(item_[100:105])
    #item_origin = item_scaler.inverse_transform(item_)

    item_ = torch.tensor(item_)
    # print(item_.shape)
    # print(item_[100:105])
    # print(item_origin[100:105])
    joblib.dump(user_scaler,'./data/user_scaler.model')
    joblib.dump(item_scaler, './data/item_scaler.model')
    print('---> user_scaler.model (generated by preprocess.py) <--- saving succeeded.')
    print('---> item_scaler.model (generated by preprocess.py) <--- saving succeeded.')
    print('2.  ------------->  preprocess down !')
    return user_, item_, user_scaler, item_scaler

if __name__ == "__main__":
    dataprocess()





