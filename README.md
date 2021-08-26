# 推荐系统
当下推荐系统主要分为两个步骤，即召回和预测。

召回是为了获取候选物品，即在所有数据中使用召回算法，获得一定数量的候选物品。

预测是对候选物品进行排序，预测其对应的点击率，即在候选物品中使用排序算法，按点击率从高到低对候选物品进行排序。

NCF 和 Conv_NCF 为召回层算法，deepFM 为排序层算法。

## ncf_
Neural Collaborative Filtering

utils.py movielens数据预处理

preprocess.py xhj数据预处理

main.py 训练模型

load_pretrain.py 加载预训练模型并测试

recall.py 加载预训练模型并预测

word2vec_train.py 训练词向量模型

word2vec.py 使用中文词向量模型映射中文

## con_ncf
Outer Product-based Neural Collaborative Filtering
## deep_fm
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

featurization.py 特征工程(to do)
## xhj_brain
mlp

preprocess.py xhj数据预处理

train.py 训练模型

model.py 模型

inference.py 预测：输入用户属性，预测高维房源值

map.py 映射：输入房源属性，映射高维房源值

fill_empty.py 数据预处理：原始数据中，空值的填充方法

test.py 测试所有功能

详情看文件夹内 readme.txt