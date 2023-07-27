######收敛到0.7
#coding=utf-8

import torch

# 检查是否可用GPU并设置为使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

from multiprocessing import Pool, Process
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import json, torch

dataset_train = []
dataset_test = []

def fecth_file(file_name, tag, a):
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_dict = json.loads(line)
            content = "Question: " + ' '.join(tmp_dict['Question'])
            content += "Answer: " + ' '.join(tmp_dict['Answer'])
            if a == 1:
                dataset_train.append([content, tag])
            else:
                dataset_test.append([content, tag])

def load_train_data():
    fecth_file("/kaggle/input/mydatas/datas/highQualityTrain.jsonl", 1, 1)
    fecth_file("/kaggle/input/mydatas/datas/lowQualityTrain.jsonl", 0, 1)

def load_test_data():
    fecth_file("/kaggle/input/mydatas/datas/highQualityTest.jsonl", 1, 2)
    fecth_file("/kaggle/input/mydatas/datas/lowQualityTest.jsonl", 0, 2)

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')#TODO:修改此处使用的预训练模型，并测试其效果
print("%%%1")
#用于对数据进行批处理和编码，并返回编码的结果
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


load_train_data()
print("%%%2")
loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)#TODO:修改此处使用的预训练模型，并测试其效果
# pretrained = BertModel.from_pretrained('bert-base-chinese')#TODO:修改此处使用的预训练模型，并测试其效果

#不训练pretrained的参数，不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Sequential( torch.nn.Conv1d(1, 16, kernel_size=3,padding=1),torch.nn.BatchNorm1d(16), torch.nn.ReLU(True) )
        self.fc2 = torch.nn.Sequential( torch.nn.Conv1d(16, 64, kernel_size=3,padding=1),torch.nn.BatchNorm1d(64), torch.nn.ReLU(True) )
        self.fc3 = torch.nn.Sequential( torch.nn.Conv1d(64, 256, kernel_size=3,padding=1),torch.nn.BatchNorm1d(256), torch.nn.ReLU(True) )
        self.fc4 = torch.nn.Sequential( torch.nn.Linear(256*768, 2), torch.nn.BatchNorm1d(2), torch.nn.ReLU(True))

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = out.last_hidden_state[:, 0]
        out = out.view(out.shape[0],-1,768)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(out.shape[0],-1)
        out = self.fc4(out)
        out = out.softmax(dim=1)
        return out


model = Model()
model = model.to(device)

print("%%%3")

#训练下游任务模型的参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2,weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()

accuracy = []

#测试
load_test_data()
def test():
    print("test begin!")
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                              batch_size=16,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print("the accurate rate is ")
    print(correct / total)
    accuracy.append(correct / total)
#TO CHECK
# criterion = criterion.to(device)
#END
for j in range(30):
    model.train()
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        #TO CHECK
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        #END
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    test()

print(accuracy)

#0.65 0.625 0.675 0.66875