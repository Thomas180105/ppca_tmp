1.py的下游任务模型由四个全连接层组成。每个全连接层都由线性层和批量归一化层组成，并使用ReLU激活函数进行非线性变换，目前可以达到0.72~0.74（80轮结果）

2.py的下游任务模型是实验性质的多层卷积神经网络，目前这个版本可以达到0.7附近的准确率（10轮左右的实验结果）

阅读1.py的时候，你可以留意以下内容

下游任务模型的定义

optimizer的定义

进行了多轮次训练
