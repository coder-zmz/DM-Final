# DM-Final

数据挖掘期末作业



`colab.ipynb` 文件分为以下几部分

- 文本特征提取，使用词袋模型、TF-IDF
  - 词袋模型参数：
    1. 忽略词频高于0.8，低于0.01的词
    2. 使用nltk库的WordNetLemmatizer进行词形还原
    3. 分词词组设置(1, 2)
  - TF-IDF使用L2范数
- 读取数据，使用torch_geometric库，创建Data对象
  - `Data(x=x, edge_index=edge_index, y=y)` x是节点特征，edge_index是边信息，y是节点标签
  - 在函数内部指定数据集位置
- 数据预处理
  - PCA降维，降至500维
  - 数据集划分，每种节点选取 40 个作为训练集，并随机选择 1000 个节点作为测试集，其余节点作为验证集
  - 孤立节点处理（待完成）
- 模型训练
  - 评价函数，计算模型在指定数据集上的所有指标
  - CORA+GCN
  - CORA+CAT
