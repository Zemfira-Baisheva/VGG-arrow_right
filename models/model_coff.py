import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights


class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 4)
        for i in self.model.parameters():
            i.requires_grad = False

        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)

# # Создайте экземпляр модели


# model = MyResNet()
# # Укажите путь к файлу, в котором сохранена модель и веса
# checkpoint_path = 'models/model_weights_coffee.pth'

# # Загрузите состояние модели и оптимизатора из файла
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
# model.load_state_dict(checkpoint)
