import imp
from random import seed
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm as tqdm
class adaptiveMLP(nn.Module):
    # TODO: implementation ：平滑处理设计
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super(adaptiveMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=bias)
        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.L2Norm(p=2, dim=1)
        # 是否针对relu函数的权重初始化
        # nn.init.kaiming_normal_(self.fc1.weight)
        # nn.init.kaiming_normal_(self.fc2.weight)


    def forward(self, x):
        # tmp = x
        x = F.relu(self.fc1(x))
        # x = (self.fc2(x)) + tmp
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = torch.abs(x)
        x = x/torch.sum(x)
        return x
    
    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1.weight)
        # reg 取其绝对值
        reg = torch.abs(reg)
        return reg
    
    def adaptive_l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0
        reg += torch.sum(self.fc1.weight**2)
        reg += torch.sum(self.fc2.weight**2)
        return reg

def adap_reweight_step(adp_model, train_loader, lambda1, notears_model, epoch_num, lrate):
    loop = tqdm.tqdm(range(epoch_num))
    for epoch in loop:
        for i, data in enumerate(train_loader):
            X = data[0]
            # W_star = W_star.to(X.device)
            with torch.no_grad():
                X_hat = notears_model(X)
            optimizer = torch.optim.Adam(adp_model.parameters(), lr=lrate, weight_decay=lambda1)
            R = X - X_hat
            R = R.to(X.device)
            reweight_list = []
            optimizer.zero_grad()
            reweight_list = adp_model(X) # FIXME: 注意这里的输入和在主函数训练的输入的一致性，要么都是R,要么都是X
            # loss 要加上了l1正则项
            loss = -0.5*torch.sum(torch.mul(reweight_list, R**2)) + lambda1*adp_model.adaptive_l2_reg()
            loss.backward()
            optimizer.step()
            loop.set_postfix(adaptive_loss=loss.item())
            # for param in adp_model.fc1.parameters():
            #     # 打印梯度
            #     print(param.grad)

    print(f'平均值：{torch.mean(reweight_list)}')
    print(f'最大值：{torch.max(reweight_list).item()}')
    print(f'最小值：{torch.min(reweight_list).item()}')
        
    return reweight_list


# 测试上述的模型
if __name__ == '__main__':
    # import TensorDataset
    import random
    import torch.utils.data as dataset
    # 创建一个3层MLP，用softmax函数保证输出是[input_size,1] 的表示概率的向量, 包括权重初始化
    X = torch.randn(800, 10)
    # 设置随机数种子函数
    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    set_random_seed(1)
    X1_data = dataset.TensorDataset(X)
    train_loader1 = dataset.DataLoader(X1_data, batch_size=100, shuffle=True)

    model = adaptiveMLP(input_size=X.shape[1], hidden_size=X.shape[1], output_size=1)
    W_star = torch.randn(10, 10)
    # 调用上述函数，进行模型的训练
    M = adap_reweight_step(model, train_loader1, lambda1=0.1, notears_model=model, epoch_num=1000, lrate = 0.001)
    print("finished")
    # print(M)
    # print(M.sum())
    pass
        
    
