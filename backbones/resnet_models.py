import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnext50_32x4d
import timm
import pretrainedmodels


class ResNetModels(nn.Module):
    def __init__(self, embedding_dim, num_classes, image_size, archi, pretrained=True, dropout=0.4):
        super(ResNetModels, self).__init__()
        self.archi = archi
        if self.archi == "resnet18":
            self.model = resnet18(pretrained)
        elif self.archi == "resnet34":
            self.model = resnet34(pretrained)
        elif self.archi == "resnet50":
            self.model = resnet50(pretrained)
        elif self.archi == "resnet101":
            self.model = resnet101(pretrained)
        elif self.archi == "resnext":
            self.model = resnext50_32x4d(pretrained)
        elif self.archi == "se_resnet50":
            self.model = torch.hub.load(
                'moskomule/senet.pytorch', 'se_resnet50', pretrained=pretrained)
        elif self.archi == "se_resnet34":
            self.model = timm.create_model('seresnet34', pretrained=pretrained)
        elif self.archi == "se_resnext50_32x4d":
            self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](
                num_classes=1000, pretrained='imagenet')

        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.embedding_dim = embedding_dim
        self.output_conv = self._get_output_conv(
            (1, 3, image_size, image_size))
        self.model.fc = nn.Linear(self.output_conv, self.embedding_dim)
        self.model.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, x):
        if self.archi not in ["se_resnet34", "se_resnext50_32x4d"]:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
        else:
            x = self.model.layer0(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.pooling_layer(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        x = self.bn(x)

        self.features = x

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        features = self.dropout(features)
        logits = self.model.classifier(features)
        return features, logits

    def _get_output_conv(self, shape):
        x = torch.rand(shape)
        if self.archi not in ["se_resnet34", "se_resnext50_32x4d"]:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
        else:
            x = self.model.layer0(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        output_conv_shape = x.size(1)
        return output_conv_shape
