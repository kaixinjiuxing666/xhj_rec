依赖包：
torch
numpy
pandas
jieba
gensim
joblib
scikit-learn
#################################################
train.py
使用方法：python train.py
文件详情：训练网络

test.py
使用方法：python test.py
文件详情：测试模型（1、输入用户属性，预测高维房源值。2、输入房源属性，输出高维房源值。）

model.py
使用方法：被调用
文件详情：网络模型

preprocess.py
使用方法：被调用
文件详情：数据预处理

map.py
使用方法：被调用
文件详情：房屋属性映射

inference.py
使用方法：被调用
文件详情：输入用户属性，预测房源属性值及其映射值

word2vec_train.py
使用方法：被调用
文件详情：训练中文词向量模型

fill_empty.py
使用方法：python fill_empty.py
文件详情：生成存储用户和房源的各个属性中位数文件 user_median_value.json, item_median_value.json

word2vec.py
使用方法：被调用
文件详情：中文映射为向量

map_server.py
房源映射服务接口
inference_server.py
预测服务接口
##################################################
./data : 

mlp.params
网络模型参数

all_ch_dict.dict
中文词向量词典

ch_all_model.tfidf
中文词向量模型

item_scaler.model
sklearn标准化模型

user_scaler.model
sklearn标准化模型
