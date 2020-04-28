import torch
import torch.nn as nn
from torchvision.models import densenet121, densenet161


class DenseNetModels(nn.Module):
    def __init__(self, embedding_dim, num_classes, image_size, archi="densenet121", pretrained=True, dropout=0):
        super(DenseNetModels, self).__init__()
        if archi == "densenet121":
            self.model = densenet121(pretrained=pretrained)
        elif archi == "densenet161":
            self.model = densenet161(pretrained=pretrained)
        self.embedding_dim = embedding_dim
        self.output_conv = self._get_output_conv(
            (1, 3, image_size, image_size))
        self.model.classifier = nn.Linear(self.output_conv, self.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.embedding_dim)
        self.model.classification_layer = nn.Linear(
            self.embedding_dim, num_classes)

    def forward(self, x):
        x = self.model.features(x)
        x = self.pooling_layer(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        x = self.bn(x)
        return x

    def _get_output_conv(self, shape):
        x = torch.rand(shape)
        x = self.model.features(x)
        output_conv_shape = x.size(1)
        return output_conv_shape

    def forward_classifier(self, x):
        features = self.forward(x)
        features = self.dropout(features)
        logits = self.model.classification_layer(features)
        return features, logits
