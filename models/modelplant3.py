# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes=30):
        super(FineTunedResNet50, self).__init__()
        # Загрузка предобученной модели densenet201
        self.model = models.densenet201(pretrained=True)

        # Заменяем последний слой на новый (с num_classes выходами)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        # Замораживаем все слои, кроме последнего
        for param in self.model.parameters():
            param.requires_grad = False

        # Размораживаем последний слой
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
# Функция для загрузки модели
def load_model(model_path='08-05-plants-v3.pt', device='cpu'):
    # Создаем модель
    model = FineTunedResNet50(num_classes=30)
    
    # Загружаем веса модели
    # Важно: если модель сохранялась с использованием GPU, а сейчас загружается на CPU, используйте map_location
    checkpoint = torch.load(model_path, map_location=device)
    
    # Загружаем веса в модель с параметром strict=False (чтобы игнорировать несоответствия, например, для замены последнего слоя)
    model.load_state_dict(checkpoint, strict=False)
    
    # Переводим модель на нужное устройство (GPU или CPU)
    model.to(device)
    
    # Переводим модель в режим оценки (для инференса)
    model.eval()
    
    return model
