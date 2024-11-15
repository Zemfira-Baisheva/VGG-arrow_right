import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_coff import MyResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from PIL import Image
import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



file_paths = [
    '/home/zemfira/Фаза_2/VGG-arrow_right/images/dark.png',
    '/home/zemfira/Фаза_2/VGG-arrow_right/images/green.png',
    '/home/zemfira/Фаза_2/VGG-arrow_right/images/light.png',
    '/home/zemfira/Фаза_2/VGG-arrow_right/images/medium.png'
]

# Проверка существования файлов и загрузка изображений
images = []
for path in file_paths:
    if os.path.exists(path):
        images.append(Image.open(path))
    else:
        st.error(f"Файл не найден: {path}")

st.write("#### Датасет Кофейные зерна")

names = ['dark', 'green', 'light', 'medium']

for i in range(0, len(images), 2):  
    cols = st.columns(2)  
    for j in range(2):
        if i + j < len(images):
            cols[j].image(images[i + j], caption=f'Класс {i + j} - {names[i + j]}', use_column_width=True, width=150)



@st.cache_resource()
def load_model():
    model = MyResNet()
    checkpoint_path = '/home/zemfira/Фаза_2/VGG-arrow_right/models/model_weights_coffee.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint)
    return model


model = load_model()

trnsfrms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      ])

uploaded_file = st.file_uploader("Загрузи изображение", type=["jpg", "jpeg", "png"])

model.eval()

def predict(img):
    image = Image.open(img).convert('RGB') 
    input_image = trnsfrms(image).unsqueeze(0).to('cpu')
    start_time = time.time()
    with torch.no_grad():
        preds = model(input_image)
    probabilities = torch.nn.functional.softmax(preds, dim=1)
    top4_prob, top4_catid = torch.topk(probabilities, 4)
    top_probs = top4_prob.squeeze().cpu().numpy()
    top_classes = top4_catid.squeeze().cpu().numpy()
    end_time = time.time()
    execution_time = end_time - start_time
    pred_class = probabilities.argmax(dim=1).item()
    st.write("#### Загруженное изображение")
    st.write(f"#### Класс - {pred_class}")
    st.image(image, caption=' ', use_column_width=True)

    st.write("#### Предсказанные классы и вероятности:")
    for i in range(4):
        st.write((f"**Класс**: {top_classes[i]}, **Вероятность**: {top_probs[i]:.3f}"))

    st.write(f"#### **Время выполнения: {execution_time:.3f} секунд**")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top_classes, top_probs)
    ax.set_xlabel('Классы')
    ax.set_ylabel('Вероятности')
    ax.set_title('Вероятности предсказанных классов')
    ax.set_xticks(top_classes)
    st.pyplot(fig)

    return pred_class



button = st.button("Вывести результаты")
if button:
    predict(uploaded_file)


