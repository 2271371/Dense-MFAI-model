import torch
import torch.nn as nn
from torchvision import models
import math
import torch.nn.functional as F
num_class = 2

def Modified_SPPLayeravg(num_levels,x,pool_type='avg_pool'):
    # num:样本数量 c:通道数 h:高 w:宽
    # num: the number of samples
    # c: the number of channels
    # h: height
    # w: width
    num, c, h, w = x.size()
    #print(x.size())
    for i in range(len(num_levels)):
        #level = i + 1

        '''
        The equation is explained on the following site:
        http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
        '''
        kernel_size = (math.ceil(h / num_levels[i]), math.ceil(w / num_levels[i]))
        stride = (math.floor(h / num_levels[i]), math.floor(w / num_levels[i]))
        pooling = (
            math.floor((kernel_size[0] * num_levels[i] - h + 1) / 2), math.floor((kernel_size[1] * num_levels[i] - w + 1) / 2))

        # update input data with padding
        zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
        x_new = zero_pad(x)

        # update kernel and stride
        h_new = 2 * pooling[0] + h
        w_new = 2 * pooling[1] + w

        kernel_size = (math.ceil(h_new / num_levels[i]), math.ceil(w_new / num_levels[i]))
        stride = (math.floor(h_new / num_levels[i]), math.floor(w_new / num_levels[i]))

        # 选择池化方式
        if pool_type == 'max_pool':
            try:
                tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            except Exception as e:
                print(str(e))
                print(x.size())
                #print(level)
        else:
            tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

        # 展开、拼接
        if (i == 0):
            x_flatten = tensor.view(num, -1)
        else:
            x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
    return x_flatten


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输入BCHW -> 输出 B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.fc(y).view(b, c, 1, 1)  # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return x * y.expand_as(x)  # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。



class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0,reduction=16):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input_features, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            SELayer(growth_rate, reduction)
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)

        return torch.cat([x, y], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class _TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet(nn.Module):
    def __init__(self, num_init_features=64, growth_rate=32, blocks=(6, 12, 24, 16), bn_size=4, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        self.layer1 = _DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transtion1 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer2 = _DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        #SELayer(growth_rate, reduction=16)
        self.transtion2 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer3 = _DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transtion3 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer4 = _DenseBlock(num_layers=blocks[3], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        #SELayer(growth_rate, reduction=16)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(num_features, 1)
        self.fc = nn.Linear(10752, 1)#7168

        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用了何凯明的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        num_levels = [4, 2, 1]
        x = self.features(x)

        x = self.layer1(x)
        x = self.transtion1(x)
        x = self.layer2(x)

        x = self.transtion2(x)
        x = self.layer3(x)
        x = self.transtion3(x)
        x = self.layer4(x)
        x = Modified_SPPLayeravg(num_levels, x)
        #print(x.size())
        #x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        #x = x.view(x.size(0), -1)
        return x




def DenseNet121():
    return DenseNet(blocks=(6, 12, 24, 16), num_classes=num_class)

def DenseNet169():
    return DenseNet(blocks=(6, 12, 32, 32), num_classes=num_class)

def DenseNet201():
    return DenseNet(blocks=(6, 12, 48, 32), num_classes=num_class)

def DenseNet264():
    return DenseNet(blocks=(6, 12, 64, 48), num_classes=num_class)

def read_densenet121():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet121(pretrained=True)
    model.to(device)
    #print(model)


def get_densenet121(flag, num_classes):
    if flag:
        net = models.densenet121(pretrained=True)
        num_input = net.classifier.in_features
        net.classifier = nn.Linear(num_input, num_classes=num_class)
    else:
        net = DenseNet121()

    return net

