from backbones.resnet_models import ResNetModels
from backbones.densenet_models import DenseNetModels


def get_model(embedding_dim,
              num_classes,
              pretrained,
              dropout,
              image_size,
              archi):

    if archi.startswith('resnet'):
        model = ResNetModels(embedding_size=embedding_dim,
                             num_classes=num_classes,
                             image_size=image_size,
                             archi=archi,
                             pretrained=pretrained,
                             dropout=dropout)

    elif archi.startswith('densenet'):
        model = DenseNetModels(embedding_size=embedding_dim,
                               num_classes=num_classes,
                               image_size=image_size,
                               archi=archi,
                               pretrained=pretrained,
                               dropout=dropout)
    return model
