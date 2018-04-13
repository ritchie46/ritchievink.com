import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        num_classes = 1000
        super(VGG, self).__init__()
        self.features = self.make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class VGGmod(nn.Module):
    def __init__(self, model, h, do):
        super(VGGmod, self).__init__()
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(25088, h),
            nn.ReLU(inplace=True),
            nn.Dropout(do),
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
            nn.Dropout(do),
            nn.Linear(h, 28)
        )

    def forward(self, x):
        x = self.features(x)

        # view is a resizing method
        x = x.view(x.size(0), -1)  # -1 means infer the shape based on the other dimension
        x = self.classifier(x)

        # area type
        a = F.sigmoid(x[:, 0])
        # curvature
        b = F.softmax(x[:, 1: 4], dim=1)
        # facilities for bicycles
        c = F.softmax(x[:, 4: 7], dim=1)
        # lane width
        d = F.softmax(x[:, 7: 10], dim=1)
        # median type
        e = F.softmax(x[:, 10: 20], dim=1)
        # number of lanes
        f = F.softmax(x[:, 20: 23], dim=1)
        # rest
        g = F.sigmoid(x[:, 23:])

        return torch.cat([a.unsqueeze(-1), b, c, d, e, f, g], dim=1)


model_pretrained = VGG()
model_pretrained.load_state_dict(model_zoo.load_url(model_urls['https://download.pytorch.org/models/vgg11_bn-6002323d.pth']))

model = VGG11mod(model_pretrained, 300, 0.5)


