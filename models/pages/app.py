import streamlit as st
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
from modelplant3 import load_model
import torch.nn.functional as F
from io import BytesIO  # Нужно для работы с байтовыми данными


st.title('Узнай растение по фото')
st.write('модель обучена на следующих растениях')

text=''' 
0.Вишня, 1. Кофейное дерево, 2. Огурец, 3. Лисий орех (Макхана), 4. Лимон

5.Оливковое дерево, 6. Просо (баджра), 7. Табак, 8. Миндаль, 9. Банан

10.Кардамон, 11. Чили, 12. Гвоздика, 13. Кокос, 14. Хлопок

15.Грамм, 16. Джовар, 17. Джут, 18. Кукуруза, 19. Горчичное масло

20.Папайя, 21. Ананас, 22. Рис, 23. Соя, 24. Сахарный тростник

25.Подсолнечник, 26. Чай, 27. Помидор, 28. Вигна-радиати (маш), 29. Пшеница
 
    '''
st.write(text)

# Функция для предсказания топ-5 меток
def get_top_k_predictions(model, image, k=5, device='cpu'):
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)  # Получаем необработанные выходные значения (logits)
    
    probabilities = F.softmax(output, dim=1)
    topk_values, topk_indices = torch.topk(probabilities, k, dim=1)
    
    return topk_values, topk_indices

# Загрузка модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model()

# Пример обработки изображения
def process_image(image_file):
    # Открываем изображение из BytesIO объекта
    image = Image.open(image_file)
    
    # Применяем необходимые трансформации
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)  # Добавляем batch размер
    
    topk_values, topk_indices = get_top_k_predictions(model, image, k=5, device=device)
    
    return topk_values, topk_indices

# Streamlit UI
st.title("Модель Densenet201")
st.write("Загрузите изображение для предсказания топ-5 меток.")

# Загрузка файла изображения
image_file = st.file_uploader("Выберите изображение", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Отображаем изображение
    st.image(image_file, caption="Загруженное изображение.", use_column_width=True)
    
    # Обрабатываем изображение
    topk_values, topk_indices = process_image(image_file)

    # Показать результаты
    st.write("Топ-5 вероятностей:", topk_values)
    st.write("Топ-5 меток:", topk_indices)