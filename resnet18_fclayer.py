import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
#from torchvision.models import ResNet
import torch.nn.functional as F
import math
# 分类数目
num_class = 2
# 各层数目
resnet18_params = [2, 2, 2, 2]
resnet34_params = [3, 4, 6, 3]
resnet50_params = [3, 4, 6, 3]
resnet101_params = [3, 4, 23, 3]
resnet152_params = [3, 8, 36, 3]


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
                #print(x.size())
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

# 定义Conv1层
def Conv1(in_planes, places, stride=2):
    #print(in_planes)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


# 浅层的残差结构
class BasicBlock(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 1,reduction=16):
        super(BasicBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        # torch.Size([1, 64, 56, 56]), stride = 1
        # torch.Size([1, 128, 28, 28]), stride = 2
        # torch.Size([1, 256, 14, 14]), stride = 2
        # torch.Size([1, 512, 7, 7]), stride = 2
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
            #SELayer(places, reduction)
        )

        # torch.Size([1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 14, 14])
        # torch.Size([1, 512, 7, 7])
        # 每个大模块的第一个残差结构需要改变步长
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.Sigmoid()

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.basicblock(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# 深层的残差结构
class Bottleneck(nn.Module):

    # 注意:默认 downsampling=False
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4, reduction=16):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 56, 56])，stride=1
            # torch.Size([1, 256, 28, 28]), stride=1
            # torch.Size([1, 512, 14, 14]), stride=1
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 64, 56, 56])，stride=1
            # torch.Size([1, 128, 28, 28]), stride=2
            # torch.Size([1, 256, 14, 14]), stride=2
            # torch.Size([1, 512, 7, 7]), stride=2
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # torch.Size([1, 256, 56, 56])，stride=1
            # torch.Size([1, 512, 28, 28]), stride=1
            # torch.Size([1, 1024, 14, 14]), stride=1
            # torch.Size([1, 2048, 7, 7]), stride=1
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
            #SELayer(places * self.expansion, reduction)
        )

        # torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7])
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.Sigmoid()

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.bottleneck(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,blocks, blockkinds, num_classes=num_class):
        super(ResNet,self).__init__()

        self.blockkinds = blockkinds
        self.conv1 = Conv1(in_planes = 3, places= 64)

        # 对应浅层网络结构
        if self.blockkinds == BasicBlock:
            self.expansion = 1
            # 64 -> 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 64 -> 128
            self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
            SELayer(128, reduction=16)
            # 128 -> 256
            self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)

            # 256 -> 512
            self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

            #self.fc = nn.Linear(512, num_classes)10752 7168
            self.fc = nn.Linear(2688, 1)

        # 对应深层网络结构
        if self.blockkinds == Bottleneck:
            self.expansion = 4
            # 64 -> 64
            self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
            # 256 -> 128
            self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
            # 512 -> 256
            self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
            # 1024 -> 512
            self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

            #self.fc = nn.Linear(2048, num_classes)
            self.fc = nn.Linear(2048, 1)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        # 初始化网络结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用了何凯明的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):

        layers = []

        # torch.Size([1, 64, 56, 56])  -> torch.Size([1, 256, 56, 56])， stride=1 故w，h不变
        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 512, 28, 28])， stride=2 故w，h变
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 1024, 14, 14])，stride=2 故w，h变
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 2048, 7, 7])， stride=2 故w，h变
        # 此步需要通过虚线分支，downsampling=True
        layers.append(self.blockkinds(in_places, places, stride, downsampling =True))

        # torch.Size([1, 256, 56, 56]) -> torch.Size([1, 256, 56, 56])
        # torch.Size([1, 512, 28, 28]) -> torch.Size([1, 512, 28, 28])
        # torch.Size([1, 1024, 14, 14]) -> torch.Size([1, 1024, 14, 14])
        # torch.Size([1, 2048, 7, 7]) -> torch.Size([1, 2048, 7, 7])
        # print("places*self.expansion:", places*self.expansion)
        # print("block:", block)
        # 此步需要通过实线分支，downsampling=False， 每个大模块的第一个残差结构需要改变步长
        for i in range(1, block):
            layers.append(self.blockkinds(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        num_levels = [4, 2, 1]
        # conv1层
        #print(x.size())
        x = self.conv1(x)   # torch.Size([1, 64, 56, 56])

        # conv2_x层
        x = self.layer1(x)  # torch.Size([1, 256, 56, 56])
        # conv3_x层
        x = self.layer2(x)  # torch.Size([1, 512, 28, 28])
        # conv4_x层
        x = self.layer3(x)  # torch.Size([1, 1024, 14, 14])
        # conv5_x层
        x = self.layer4(x)  # torch.Size([1, 2048, 7, 7])
        x = Modified_SPPLayeravg(num_levels, x)
        #x = self.avgpool(x) # torch.Size([1, 2048, 1, 1]) / torch.Size([1, 512])
        #x = x.view(x.size(0), -1)   # torch.Size([1, 2048]) / torch.Size([1, 512])
        #x = self.fc(x)      # torch.Size([1, 5])

        #x = x[:,0]
        return x
        #return x

def ResNet18():
    return ResNet(resnet18_params, BasicBlock)

def ResNet34():
    return ResNet(resnet34_params, BasicBlock)

def ResNet50():
    return ResNet(resnet50_params, Bottleneck)

def ResNet101():
    return ResNet(resnet101_params, Bottleneck)

def ResNet152():
    return ResNet(resnet152_params, Bottleneck)
