from torch_nn_train_toolkit import *

def load_data_mnist(batch_size, resize=None):
    """下载mnist数据集，然后将其加载到内存中。"""
    trans = [    transforms.Resize(32),
                 transforms.CenterCrop(28),
                 transforms.ToTensor(),
                 transforms.Normalize((0.1037,),(0.3081,)), #正则化，模型出现过拟合现象时，降低模型复杂度,
                 ]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root="./data", 
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.MNIST(root="./data",
                                                   train=False,
                                                  transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


class Res_block(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 
                                   kernel_size=1,stride=strides,bias=False)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y +=X     # 跳跃连接
        return self.relu(Y)

class ResNet(nn.Module):
    """建造Res-n网络的封装函数"""
    def __init__(self, layer_sum, fig_chan=3, num_classes=10):
        """layers_sum是神经网络的总层数
            fig_chan是输入图像通道数；num_classes是分类输出种类数"""
        super(ResNet, self).__init__()
        self.n=(layer_sum-2)//8 # He et.al的网络定义，加上全连接层总层数为6n+2
        self.in_channels = 64
        # 开头的残差块前网络
        self.conv1 = nn.Conv2d(fig_chan, self.in_channels, kernel_size=3,
                            stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        # 残差块网络定义
        self.layer1 = self._make_layer(64, 64, self.n,first_block=True)
        self.layer2 = self._make_layer(64, 128, self.n)
        self.layer3 = self._make_layer(128, 256, self.n)
        self.layer4 = self._make_layer(256, 512, self.n)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Dense = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    # 搭建指定输入通道的残差块
    def _make_layer(self, in_chans, out_chans, num_blocks, first_block=False):
        blk=[]
        for i in range(num_blocks):
            if i==0 and first_block:
                blk.append(Res_block(in_chans, out_chans))
            elif i==0:
                blk.append(Res_block(in_chans, out_chans, use_1x1conv=True,strides=2))
            else:
                blk.append(Res_block(out_chans, out_chans))
        return nn.Sequential(*blk)

    def forward(self, x):
        # 开头的残差块前网络
        out = self.relu(self.bn1(self.conv1(x)))
        # 残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全局池化，全连接输出
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.Dense(out)
        out = self.softmax(out)

        return out

model=ResNet(18, num_classes=10, fig_chan=1)

#print(model)

total_epoch=0

epoch=0

lr, num_epochs, batch_size = 0.0002, 10, 128

optimizer= torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

train_iter, test_iter = load_data_mnist(batch_size)

loss = nn.CrossEntropyLoss()

model,epoch,success=load_model_params(model,path="resnet.ckpt")

train_modl(model, train_iter, test_iter, num_epochs, loss, lr, device=try_gpu(), init=not success, optimizer=optimizer)

total_epoch=epoch+num_epochs

save_model(model,epoch=total_epoch,path="resnet.ckpt")



