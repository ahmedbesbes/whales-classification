import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101


class ResNetModels(nn.Module):
    def __init__(self, embedding_size, num_classes, image_size, archi, pretrained=True, dropout=0.4):
        super(ResNetModels, self).__init__()

        if archi == "resnet18":
            self.model = resnet18(pretrained)
        elif archi == "resnet34":
            self.model = resnet34(pretrained)
        elif archi == "resnet50":
            self.model = resnet50(pretrained)
        elif archi == "resnet101":
            self.model = resnet101(pretrained)

        self.embedding_size = embedding_size
        self.output_conv = self._get_output_conv(
            (1, 3, image_size, image_size))
        self.model.fc = nn.Linear(self.output_conv, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.model.fc(x)

        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features*alpha

        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)

        return res

    def _get_output_conv(self, shape):
        x = torch.rand(shape)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        output_conv_shape = x.size(1)
        return output_conv_shape
