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


heatmap = Image.open('/home/zemfira/Фаза_2/VGG-arrow_right/images/heatmap.png')
curves = Image.open('/home/zemfira/Фаза_2/VGG-arrow_right/images/learning_curves.png')
plants = Image.open('/home/zemfira/Фаза_2/VGG-arrow_right/images/plants.png')


st.write("## Информация о модели")

st.write("#### была использована модель ResNet50, время обучения - 3 эпохи")

st.image(curves, caption=' ', use_column_width=True)

st.write("#### Матрица ошибок")

st.image(heatmap, caption='f1_score = 0,992', use_column_width=True)

st.image(curves, caption=' ', use_column_width=True)

st.write("#### Распознавание растений")

st.image(plants, caption='f1_score = 0,992', use_column_width=True)
