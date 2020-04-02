import torch
import torch.nn as nn
from torchvision.models import densenet121


class DenseNetModels(nn.Module):
    def __init__(self, embedding_dim, num_classes, image_size, archi="densenet121", pretrained=True, dropout=0.4, alpha=10, gap=1):
        super(DenseNetModels, self).__init__()
        if archi == "densenet121":
            self.model = densenet121(pretrained=pretrained)
        self.embedding_dim = embedding_dim
        self.gap = gap
        self.output_conv = self._get_output_conv(
            (1, 3, image_size, image_size))
        self.model.fc = nn.Linear(self.output_conv, self.embedding_dim)
        self.model.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.features(x)
        if self.gap == 1:
            x = self.pooling_layer(x)
            x = x.view(x.size(0), -1)

        elif self.gap == 0:
            x = x.view(x.size(0), -1)
            x = self.dropout(x)

        x = self.model.fc(x)
        self.features = self.l2_norm(x)

        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        self.features = self.features * self.alpha
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

    def _get_output_conv(self, shape):
        x = torch.rand(shape)
        x = self.model.features(x)
        if self.gap == 1:
            output_conv_shape = x.size(1)
        elif self.gap == 0:
            x = x.view(x.size(0), -1)
            output_conv_shape = x.size(1)
        return output_conv_shape
