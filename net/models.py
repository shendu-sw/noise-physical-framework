import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class _Memory_Block(nn.Module):        
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        
        self.c = hdim
        self.k = kdim
        
        self.moving_average_rate = moving_average_rate
        
        self.units = nn.Embedding(kdim, hdim) # 1024*400
                
    def update(self, x, score, m=None):
        '''
            x: (n, self.c)
            e: (self.k, self.c)   self.c = hdim
            score: (n, self.k)    self.k = kdim
        '''
        if m is None:
            m = self.units.weight.data # 1024*400
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1] # (n, )
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype) # (n, k)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot # (c, k) 400*1024
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        ss = embed_mean.sum(1).unsqueeze(1)
        #print('1:',ss.size())
        param_diversity = torch.sub(embed_mean, ss) * 2/self.k/(self.k-1)
        #print(param_diversity.size())
        new_data = m * self.moving_average_rate + (embed_mean+param_diversity).t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data

    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''
        
        b, c = x.size()        # b表示batch大小， c表示维度
        assert c == self.c        
        k, c = self.k, self.c  #k表示base noise的number
        
        #x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)   # batch*400
        
        m = self.units.weight.data # (k, c)  1024*400
                
        xn = F.normalize(x, dim=1) # (n, c) # batch *400
        mn = F.normalize(m, dim=1) # (k, c) # 1024*400
        score = torch.matmul(xn, mn.t()) # (n, k) # batch * 1024
        
        if update_flag:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1) # (k, c)
            score = torch.matmul(xn, mn.t()) # (n, k)

        '''soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m) # (n, c)'''
        #out = out.view(b, c).permute(0, 3, 1, 2)

        out = torch.matmul(score, m)
        tend = torch.norm(out, dim=1) # batch*1
        tend_1 = torch.norm(x, dim=1)  # batch*1
        param = torch.div(tend_1, tend).unsqueeze(1)  # batch*1
        #print(param.size())
        #print(score.size())
        norm_label = torch.mul(param, score)
        out = torch.matmul(norm_label, m)

        return out, score


class D3_Net_Memory(nn.Module):
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def __init__(self, num_classes, in_channels=1, patch_size=5):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels

        if patch_size==3:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=1, padding=1)
        else:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=1, padding=0)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), dilation=1, stride=(2, 1, 1), padding=(1, 0, 0))
        # self.pool1 = nn.Conv3d(20, 20, (3, 3, 3), dilation=1, stride=(2, 2, 2), padding=(1, 0, 0))
        #self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), dilation=1, stride=(1, 1, 1), padding=(1, 0, 0))
        #self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), dilation=1, stride=(2, 1, 1), padding=(1, 0, 0))
        #self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), dilation=1, stride=(1, 1, 1), padding=(1, 0, 0))
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=1, stride=(2, 1, 1), padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()
        #print('self.features_size', self.features_size)   # 665
        self.feature = nn.Linear(self.features_size, 400)

        self.feature1 = nn.Linear(400, 400)

        self.memory = _Memory_Block(400, 2048, 0.99)

        self.feature_up = nn.Linear(400, 400)

        self.cls = nn.Linear(400, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        #self.fc = nn.Linear(self.features_size, num_classes)
        self.fc = nn.Linear(400, num_classes)

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):

        x = F.relu(self.conv1(x))
        #print(1)

        x = self.pool1(x)
        #print(2)
        x = F.relu(self.conv2(x))
        #print(3)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print('4:',x.size())

        #print('3:',x.size())
        x = x.view(-1, self.features_size)
        # print('5:',x.size())
        x = self.feature(x)

        
        
        fea = x

        xx = self.feature1(x)

        out1 = fea - xx

        x, _ = self.memory(xx)

        x = self.feature_up(x)

        out_feature = fea - x

        x = self.cls(out_feature)

        x1 = self.cls(out1)

        return x#, out_feature
        '''
        out_feature = x
        x = self.fc(x)

        return x, out_feature
        '''

class D3_CNN(nn.Module):
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def __init__(self, num_classes, in_channels=1, patch_size=5):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels

        if patch_size==3:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=1, padding=1)
        elif patch_size==1:
            self.conv1 = nn.Conv3d(1, 20, (3, 1, 1), stride=(1, 1, 1), dilation=1, padding=0)
        else:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=1, padding=0)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), dilation=1, stride=(2, 1, 1), padding=(1, 0, 0))
        #self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv3d(20, 35, (3, 1, 1), dilation=1, stride=(1, 1, 1), padding=(1, 0, 0))
        #self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), dilation=1, stride=(2, 1, 1), padding=(1, 0, 0))
        #self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), dilation=1, stride=(1, 1, 1), padding=(1, 0, 0))
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=1, stride=(2, 1, 1), padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()
        #print('self.features_size', self.features_size)   # 665
        self.feature = nn.Linear(self.features_size, 400)

        self.feature1 = nn.Linear(400, 400)

        self.memory = _Memory_Block(400, 1024, 0.99)

        self.feature_up = nn.Linear(400, 400)

        self.cls = nn.Linear(400, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        #self.fc = nn.Linear(self.features_size, num_classes)
        self.fc = nn.Linear(400, num_classes)

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
            #print('ss:',x.size())
        return t * c * w * h

    def forward(self, x):

        x = F.relu(self.conv1(x))
        #print(1)

        x = self.pool1(x)
        #print(2)
        x = F.relu(self.conv2(x))
        #print(3)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print('4:',x.size())

        #print('3:',x.size())
        x = x.view(-1, self.features_size)
        # print('5:',x.size())
        x = self.feature(x)

        out_feature = x

        x = self.cls(x)

        return x#, out_feature


if __name__ == "__main__":

    model = D3_Net_Memory(103, 9)
    print(model)
