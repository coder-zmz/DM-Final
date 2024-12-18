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
  
  - 数据可视化
  - PCA降维，降至500维
  - 数据集划分，每种节点选取 40 个作为训练集，并随机选择 1000 个节点作为测试集，其余节点作为验证集
  - 孤立节点处理（待完成）
  
- 模型训练
  - 评价函数，计算模型在指定数据集上的所有指标

    - Args:

      ​    model: 模型

      ​    data: 数据对象

      ​    mask: 用于选择节点的掩码，例如 data.train_mask 或 data.test_mask

    - Returns:

         一个包含所有指标的字典

  - 训练函数，训练模型并评估其在训练集和测试集上的性能

    - Args:

      ​    model: PyTorch 模型

      ​    criterion: 损失函数

      ​    optimizer: 优化器

      ​    data: PyTorch Geometric 数据对象

      ​    num_epochs: 训练轮数，默认200

      ​    patience: 控制早停，即连续多少个 epoch 验证集损失没有下降就停止训练，默认10

      ​    scheduler: 学习率调度器(可选)
    - Returns:

      ​    train_metrics_df: 存储训练集上评价指标的 Pandas DataFrame

      ​    test_metrics_df: 存储测试集上评价指标的 Pandas DataFrame

  - 分类结果可视化函数，使用 t-SNE 可视化嵌入向量

  - CORA+GCN

    - 默认，测试集准确率0.846

      | 隐藏层维度 | 学习率 | 权重衰减 | dropout | 层数 | 激活函数 |
      | ---------- | ------ | -------- | ------- | ---- | -------- |
      | 16         | 0.01   | 5e-4     | 0.5     | 2    | ReLU     |

    - 优化，测试集准确率0.859

      | 隐藏层维度 | 学习率                   | 权重衰减 | dropout | 层数 | 激活函数 |
      | ---------- | ------------------------ | -------- | ------- | ---- | -------- |
      | 64         | 0.1，loss不变时降低0.5倍 | 5e-3     | 0.3     | 2    | ReLU     |

  - CORA+CAT
