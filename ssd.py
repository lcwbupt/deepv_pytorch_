import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, deepv
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = deepv
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        # self.L2Norm = L2Norm(128, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":          
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),      # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
    '648': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'C',
            128, 128, 128],
    '768': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
    '648': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 256],
    '768': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
    '648': [4, 6, 6, 4, 4, 8, 4],
    '768': [2, 4, 4, 3, 3, 3],
}

def build_ssd(phase, size=768, num_classes=5):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
#     if size != 300:
#         print("ERROR: You specified size " + repr(size) + ". However, " +
#               "currently only SSD300 (size=300) is supported!")
#         return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)


class SSD_ResNet_18(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_ResNet_18, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (voc, deepv)[num_classes == 5]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.resnet18 = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
#         self.L2Norm = L2Norm(512, 20)
        self.L2Norm = L2Norm(128, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(4):
            x = self.resnet18[k](x)
        #layer1
        residual = x
        for k in range(4, 9):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[9](x)
        
        residual = x
        for k in range(10, 15):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[15](x)
        #layer2
        residual = x
        for k in range(22, 24):
            residual=self.resnet18[k](residual)
        for k in range(16, 21):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[21](x)
        
        residual = x
        for k in range(24, 29):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[29](x)
        #layer3
        residual = x
        for k in range(36, 38):
            residual=self.resnet18[k](residual)
        for k in range(30, 35):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[35](x)
        
        residual = x
        for k in range(38, 43):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[43](x)
        sources.append(x)
        #layer4
        residual = x
        for k in range(50, 52):
            residual=self.resnet18[k](residual)
        for k in range(44, 49):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[49](x)
        
        residual = x
        for k in range(52, 57):
            x = self.resnet18[k](x)
        x+=residual
        x=self.resnet18[57](x)
        sources.append(x)

#         s = self.L2Norm(x)
#         sources.append(s)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def resnet18():
    
    layers = []
    layers += [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)]
    layers += [nn.BatchNorm2d(64)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    
    #layer1
    conv1 = conv3x3(64, 64, 1)
    bn1 = nn.BatchNorm2d(64)
    relu1 = nn.ReLU(inplace=True)
    conv2 = conv3x3(64, 64)
    bn2 = nn.BatchNorm2d(64)
    relu2 = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2]
    conv1 = conv3x3(64, 64, 1)
    bn1 = nn.BatchNorm2d(64)
    relu1 = nn.ReLU(inplace=True)
    conv2 = conv3x3(64, 64)
    bn2 = nn.BatchNorm2d(64)
    relu2 = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2]
    #layer2  
    conv1 = conv3x3(64, 128, 2)
    bn1 = nn.BatchNorm2d(128)
    relu1 = nn.ReLU(inplace=True)
    conv2 = conv3x3(128, 128)
    bn2 = nn.BatchNorm2d(128)
    conv_downsample = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
    bn_downsample = nn.BatchNorm2d(128)
    relu2 = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2,conv_downsample,bn_downsample]
    conv1 = conv3x3(128, 128, 1)
    bn1 = nn.BatchNorm2d(128)
    relu1 = nn.ReLU(inplace=True)
    conv2 = conv3x3(128, 128)
    bn2 = nn.BatchNorm2d(128)
    relu2 = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2]
    #layer3
    conv1 = conv3x3(128, 256, 2)
    bn1 = nn.BatchNorm2d(256)
    relu1 = nn.ReLU(inplace=True)
    conv2 = conv3x3(256, 256)
    bn2 = nn.BatchNorm2d(256)
    conv_downsample = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
    bn_downsample = nn.BatchNorm2d(256)
    relu2 = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2,conv_downsample,bn_downsample]
    conv1 = conv3x3(256, 256, 1)
    bn1 = nn.BatchNorm2d(256)
    relu1 = nn.ReLU(inplace=True)
    conv2 = conv3x3(256, 256)
    bn2 = nn.BatchNorm2d(256)
    relu2 = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2]
    #layer4
    conv1 = conv3x3(256, 512, 2)
    bn1 = nn.BatchNorm2d(512)
    relu = nn.ReLU(inplace=True)
    conv2 = conv3x3(512, 512)
    bn2 = nn.BatchNorm2d(512)
    conv_downsample = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
    bn_downsample = nn.BatchNorm2d(512)
    relu = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2,conv_downsample,bn_downsample]
    conv1 = conv3x3(512, 512, 1)
    bn1 = nn.BatchNorm2d(512)
    relu = nn.ReLU(inplace=True)
    conv2 = conv3x3(512, 512)
    bn2 = nn.BatchNorm2d(512)
    relu = nn.ReLU(inplace=True)
    layers += [conv1,bn1,relu1,conv2,bn2,relu2]
#     layers += nn.AvgPool2d(7, stride=1)
#     self.layer1 = self._make_layer(block, 64, layers[0])
#     self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#     self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#     self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            
    return layers

def multibox_resnet18(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [41, 55]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def build_ssd_resnet18(phase, size=648, num_classes=5):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
#     if size != 300:
#         print("ERROR: You specified size " + repr(size) + ". However, " +
#               "currently only SSD300 (size=300) is supported!")
#         return
    
    base_, extras_, head_ = multibox_resnet18(resnet18(),
                                     add_extras(extras[str(size)], 512),
                                     mbox[str(size)], num_classes)
    return SSD_ResNet_18(phase, size, base_, extras_, head_, num_classes)

if __name__ == '__main__':
#     net = resnet18()
    net = vgg(base[str(648)], 3)
    for k, v in enumerate(net):
        print(k, v)