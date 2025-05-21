# -*- coding: utf-8 -*-
import os
# Устанавливаем переменную окружения для обхода ошибки OpenMP (может быть полезно на некоторых системах)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

# --- Конфигурация страницы Streamlit (должна быть первым вызовом st) ---
st.set_page_config(
    page_title="Система оценки состояния угодий",
    page_icon="🌿",  # Добавлен значок
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:your_email@example.com', # Укажите ваш email
        'Report a bug': "mailto:your_email@example.com", # Укажите ваш email
        'About': "### Система оценки состояния угодий\nВерсия 1.1\n\nПриложение для анализа состояния угодий, расчета спектральных индексов и генерации отчетов."
    }
)

# --- Импорты ---
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ExifTags, ImageOps # Добавлен ImageOps для коррекции ориентации
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import io
# import base64 # Больше не используется напрямую
from streamlit_option_menu import option_menu
from datetime import datetime
from fpdf import FPDF
import math # Для проверки nan/inf

# --- Константа для пути к шрифту ---
# Укажите правильный путь к вашему TTF файлу (например, DejaVuSans.ttf)
# Поместите файл шрифта в ту же папку, что и скрипт, или укажите полный путь.
FONT_PATH = 'DejaVuSans.ttf'
# ----------------------------------

# --- Кастомный CSS ---
def set_css():
    st.markdown(
        """
        <style>
        /* --- Базовые стили --- */
        .stApp {
            /* Фоновое изображение или цвет */
             /* background-image: url(https://ваш_url/background.jpg); */
             background-color: #e8f5e9; /* Светло-зеленый фон */
             background-size: cover;
             font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }

        /* --- Заголовки --- */
        h1, h2, h3 {
            text-align: center;
            font-weight: 600;
        }
        h1 { /* Главный заголовок */
            color: #1b5e20; /* Темно-зеленый */
            text-shadow: 1px 1px 2px #a5d6a7; /* Легкая тень */
        }
        h3 { /* Подзаголовки */
            color: #388e3c; /* Средне-зеленый */
        }
        .stTabs [data-baseweb="tab-list"] {
             gap: 24px; /* Увеличить расстояние между табами */
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #c8e6c9; /* Фон таба */
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
             background-color: #a5d6a7; /* Фон выбранного таба */
        }

        /* --- Боковая панель --- */
        .css-1d391kg { /* Селектор может измениться в будущих версиях Streamlit */
            background-color: rgba(255, 255, 255, 0.9) !important; /* Белый с прозрачностью */
            border-right: 2px solid #a5d6a7; /* Зеленая граница справа */
        }

        /* --- Основной блок --- */
        .main .block-container {
             padding-top: 2rem;
             padding-bottom: 2rem;
             padding-left: 3rem;
             padding-right: 3rem;
        }
        .stBlock { /* Общий стиль для блоков внутри */
             background-color: rgba(255, 255, 255, 0.85); /* Белый с прозрачностью */
             backdrop-filter: blur(5px);
             border-radius: 10px;
             padding: 20px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
             margin-bottom: 1rem; /* Отступ снизу */
        }

        /* --- Кнопки --- */
        .stButton button {
            background-color: #4CAF50 !important; /* Зеленый */
            color: white !important;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            padding: 0.6em 1.2em;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #388E3C !important; /* Темнее при наведении */
        }
        .stDownloadButton button {
             background-color: #0277bd !important; /* Синий для скачивания */
        }
         .stDownloadButton button:hover {
              background-color: #01579b !important;
         }

        /* --- Другие элементы --- */
        .stFileUploader label {
            font-weight: bold;
            color: #1b5e20;
        }
        .stSelectbox label, .stNumberInput label, .stSlider label {
            font-weight: bold;
            color: #388e3c;
        }
        .stAlert { /* Уведомления */
             border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_css()

# --- Модель ИИ (Простой пример) ---
# --- Модель ИИ (Совместимая с model.pth) ---
class PlantClassifier(nn.Module):
    def __init__(self):
        super(PlantClassifier, self).__init__()
        # Определяем слои напрямую, как ожидает model.pth
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Рассчитываем размер входа для полносвязного слоя:
        # Исходный размер 128x128
        # После conv1 -> 128x128 (padding=1, stride=1)
        # После pool1 -> 64x64
        # После conv2 -> 64x64 (padding=1, stride=1)
        # После pool2 -> 32x32
        # Размер = 64 канала * 32 * 32
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2) # 2 выходных класса

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Преобразуем выход сверточных слоев в вектор
        x = x.view(x.size(0), -1) # x.view(-1, 64 * 32 * 32) тоже сработает
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Функции расчета индексов ---
# Глобальные переменные для индексов каналов (будут установлены в main)
red_index, green_index, blue_index, nir_index = 0, 1, 2, None

def safe_divide(numerator, denominator, epsilon=1e-7):
    """Безопасное деление для избежания деления на ноль."""
    return numerator / (denominator + epsilon)

# --- Индексы, требующие NIR ---
def calculate_ndvi(image):
    if nir_index is None: return np.full(image.shape[:2], np.nan) # Возвращаем NaN если нет NIR
    nir = image[:, :, nir_index].astype(np.float32)
    red = image[:, :, red_index].astype(np.float32)
    return safe_divide(nir - red, nir + red)

def calculate_gndvi(image):
    if nir_index is None: return np.full(image.shape[:2], np.nan)
    nir = image[:, :, nir_index].astype(np.float32)
    green = image[:, :, green_index].astype(np.float32)
    return safe_divide(nir - green, nir + green)

def calculate_endvi(image):
    if nir_index is None: return np.full(image.shape[:2], np.nan)
    nir = image[:, :, nir_index].astype(np.float32)
    green = image[:, :, green_index].astype(np.float32)
    blue = image[:, :, blue_index].astype(np.float32)
    return safe_divide((nir + green) - (2 * blue), (nir + green) + (2 * blue))

def calculate_cvi(image):
    if nir_index is None: return np.full(image.shape[:2], np.nan)
    nir = image[:, :, nir_index].astype(np.float32)
    red = image[:, :, red_index].astype(np.float32)
    green = image[:, :, green_index].astype(np.float32)
    # Добавляем epsilon к green в знаменателе для предотвращения деления на 0
    return safe_divide(nir, green) * safe_divide(red, green)

def calculate_osavi(image):
    if nir_index is None: return np.full(image.shape[:2], np.nan)
    nir = image[:, :, nir_index].astype(np.float32)
    red = image[:, :, red_index].astype(np.float32)
    # Значение 0.16 - стандартный коэффициент для OSAVI
    return safe_divide(1.5 * (nir - red), nir + red + 0.16)

# --- Индексы, работающие с RGB ---
def calculate_vari(image):
    red = image[:, :, red_index].astype(np.float32)
    green = image[:, :, green_index].astype(np.float32)
    blue = image[:, :, blue_index].astype(np.float32)
    return safe_divide(green - red, green + red - blue)

def calculate_exg(image):
    red = image[:, :, red_index].astype(np.float32)
    green = image[:, :, green_index].astype(np.float32)
    blue = image[:, :, blue_index].astype(np.float32)
    # ExG (Excess Green) - простой индекс
    return 2 * green - red - blue

def calculate_gli(image):
    red = image[:, :, red_index].astype(np.float32)
    green = image[:, :, green_index].astype(np.float32)
    blue = image[:, :, blue_index].astype(np.float32)
    numerator = 2 * green - red - blue
    denominator = 2 * green + red + blue
    return safe_divide(numerator, denominator)

def calculate_rgbvi(image):
    red = image[:, :, red_index].astype(np.float32) + 1e-7 # Добавим эпсилон сразу
    green = image[:, :, green_index].astype(np.float32) + 1e-7
    blue = image[:, :, blue_index].astype(np.float32) + 1e-7
    numerator = (green ** 2) - (red * blue)
    denominator = (green ** 2) + (red * blue)
    return safe_divide(numerator, denominator)

# --- Общая функция расчета индекса ---
def calculate_vegetation_index(image_array, index_name):
    """Рассчитывает выбранный вегетационный индекс."""
    index_functions = {
        'NDVI': calculate_ndvi,
        'GNDVI': calculate_gndvi,
        'ENDVI': calculate_endvi,
        'CVI': calculate_cvi,
        'OSAVI': calculate_osavi,
        'VARI': calculate_vari,
        'ExG': calculate_exg,
        'GLI': calculate_gli,
        'RGBVI': calculate_rgbvi,
    }
    func = index_functions.get(index_name)
    if func:
        result = func(image_array)
        # Заменяем бесконечности на NaN
        result[np.isinf(result)] = np.nan
        return result
    else:
        st.error(f"Неизвестный индекс: {index_name}")
        return None

# --- Функции для работы с моделью и изображением ---
@st.cache_resource # Кэшируем загруженную модель
def load_model(model_path='model.pth'):
    """Загружает модель PyTorch."""
    try:
        model = PlantClassifier()
        # Загружаем на CPU по умолчанию
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval() # Переводим модель в режим оценки
        return model
    except FileNotFoundError:
        st.error(f"Ошибка: Файл модели '{model_path}' не найден. Убедитесь, что он находится в той же папке, что и скрипт.")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

def predict_image(image_pil_rgb, model, transform):
    """Предсказывает класс изображения с помощью модели."""
    if model is None:
        return "Модель не загружена"

    try:
        # Применяем трансформации (изменение размера, тензор, нормализация если нужна)
        input_tensor = transform(image_pil_rgb).unsqueeze(0) # Добавляем batch измерение

        with torch.no_grad(): # Отключаем расчет градиентов
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, 1)

        # Определяем имена классов
        class_names = [
            "Удовлетворительно (наблюдается засуха или стресс)", # Индекс 0
            "Оптимально (хорошие условия полива)"                # Индекс 1
        ]
        predicted_class_name = class_names[predicted_class_index.item()]
        confidence_percent = confidence.item() * 100

        # Возвращаем имя класса и уверенность
        return f"{predicted_class_name} (Уверенность: {confidence_percent:.1f}%)"

    except Exception as e:
        st.error(f"Ошибка во время предсказания: {e}")
        return "Ошибка предсказания"

def get_image_metadata(image):
    """Извлекает EXIF метаданные из изображения."""
    metadata = {}
    try:
        exif_data = image.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                metadata[tag] = value
    except Exception:
        # Иногда getexif может вызвать ошибку для некоторых форматов
        pass
    return metadata

def correct_image_orientation(image):
    """Корректирует ориентацию изображения на основе EXIF данных."""
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
         # Если не удается прочитать EXIF или повернуть
         pass
    return image

# --- Функции расчета GSD и площади ---
def calculate_gsd(flight_altitude, focal_length_mm, sensor_width_mm, sensor_height_mm, image_width_px, image_height_px):
    """Рассчитывает Ground Sampling Distance (GSD) в метрах/пиксель."""
    if focal_length_mm <= 0 or image_width_px <= 0 or image_height_px <= 0:
        return 0, 0
    # Переводим фокусное расстояние в метры
    focal_length_m = focal_length_mm / 1000.0
    # Переводим размеры сенсора в метры
    sensor_width_m = sensor_width_mm / 1000.0
    sensor_height_m = sensor_height_mm / 1000.0

    gsd_width_m_px = (flight_altitude * sensor_width_m) / (focal_length_m * image_width_px)
    gsd_height_m_px = (flight_altitude * sensor_height_m) / (focal_length_m * image_height_px)
    return gsd_width_m_px, gsd_height_m_px

def calculate_image_area(gsd_width, gsd_height, image_width, image_height):
    """Рассчитывает площадь покрытия изображения в кв. метрах и размеры на земле."""
    if gsd_width <= 0 or gsd_height <= 0:
        return 0, 0, 0
    ground_width_m = gsd_width * image_width
    ground_height_m = gsd_height * image_height
    area_m2 = ground_width_m * ground_height_m
    return area_m2, ground_width_m, ground_height_m

# --- Функция для интерактивной карты ---
def interactive_map(center_coords=[57.153033, 65.534328], zoom=5):
    """Отображает интерактивную карту Folium."""
    st.subheader("Интерактивная карта региона")
    st.caption("Кликните на карту, чтобы выбрать координаты (не используется для расчетов в текущей версии).")

    map_object = folium.Map(location=center_coords, zoom_start=zoom, tiles="CartoDB positron") # Светлая тема карты
    # Добавляем возможность рисования (если нужно)
    # draw = folium.plugins.Draw()
    # draw.add_to(map_object)

    # Отображаем карту в Streamlit
    st_data = st_folium(map_object, width=700, height=400)

    # Показываем координаты последнего клика (если был)
    if st_data and st_data.get("last_clicked"):
        coords = st_data['last_clicked']
        st.write(f"Координаты последнего клика: {coords['lat']:.5f}, {coords['lng']:.5f}")

# --- Класс для генерации PDF отчета ---
class PDF(FPDF):
    def header(self):
        if not hasattr(self, 'font_added') or not self.font_added:
            return # Не добавлять хедер, если шрифт не загружен

        self.set_font('DejaVu', 'B', 14) # Крупнее шрифт заголовка
        self.cell(0, 10, 'Отчет об оценке состояния угодий', 0, 1, 'C')
        self.set_font('DejaVu', '', 9) # Мельче шрифт даты
        self.cell(0, 8, f'Дата генерации: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(8) # Увеличить отступ после заголовка

    def footer(self):
        if not hasattr(self, 'font_added') or not self.font_added:
            return # Не добавлять футер, если шрифт не загружен

        self.set_y(-15) # Позиция 1.5 см от низа
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}/{{nb}}', 0, 0, 'C') # {{nb}} будет заменено на общее число страниц

    def chapter_title(self, title):
        """Добавляет заголовок раздела."""
        if not hasattr(self, 'font_added') or not self.font_added: return
        self.set_font('DejaVu', 'B', 12)
        self.set_fill_color(230, 240, 230) # Светло-зеленый фон для заголовка раздела
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)
        self.set_font('DejaVu', '', 11) # Возвращаем обычный шрифт

    def chapter_body(self, content_dict=None, text_list=None):
        """Добавляет содержимое раздела (ключ-значение или список строк)."""
        if not hasattr(self, 'font_added') or not self.font_added: return
        self.set_font('DejaVu', '', 10) # Чуть мельче шрифт для содержимого
        if content_dict:
            for key, value in content_dict.items():
                 # Используем multi_cell для автоматического переноса строк
                 self.multi_cell(0, 6, f'**{key}:** {value}') # Жирный ключ
                 self.ln(1) # Небольшой отступ
        if text_list:
            for item in text_list:
                 self.multi_cell(0, 6, f'- {item}')
                 self.ln(1)
        self.ln(4) # Отступ после раздела

    def add_image_from_buffer(self, img_buffer, title, max_width=180):
        """Добавляет изображение из буфера с заголовком."""
        if not hasattr(self, 'font_added') or not self.font_added: return
        self.chapter_title(title)
        try:
            # Определяем размер изображения
            img_pil = Image.open(img_buffer)
            img_width_px, img_height_px = img_pil.size
            aspect_ratio = img_height_px / img_width_px if img_width_px > 0 else 1

            # Рассчитываем ширину и высоту в мм для PDF
            display_width = min(img_width_px * 0.264583, max_width) # Конвертируем пиксели в мм (примерно) и ограничиваем макс. шириной
            display_height = display_width * aspect_ratio

            # Центрируем изображение
            x_pos = (self.w - display_width) / 2
            self.image(img_buffer, x=x_pos, w=display_width, h=display_height, type='PNG') # Указываем тип явно
            self.ln(display_height + 5) # Отступ после изображения
        except Exception as e:
            self.set_text_color(255, 0, 0)
            self.multi_cell(0, 5, f'Не удалось вставить изображение "{title}": {e}')
            self.set_text_color(0, 0, 0)
            self.ln(5)

# --- Функция генерации PDF ---
def generate_pdf_report(
    original_image_buffer, # Буфер с исходным изображением
    final_assessment,      # Итоговая оценка (может включать инфо об индексе)
    assessment_details,    # Детали оценки (средний индекс)
    selected_index,
    mean_value_original,   # Среднее ненормализованное значение
    index_map_buffer,      # Буфер с картой индекса
    recommendations,
    area_ha, ground_width, ground_height, area_km2,
    flight_params,
    dryness_threshold_normalized, # Порог нормализованного индекса
    dryness_area_ha, dryness_percent,
    dryness_additive_calc # Строка с расчетом добавок
    ):
    """Генерирует PDF отчет с результатами анализа."""

    pdf = PDF('P', 'mm', 'A4') # Портретная ориентация, мм, A4
    pdf.font_added = False # Флаг, что шрифт еще не добавлен

    # --- Проверка и добавление шрифта ---
    if not os.path.exists(FONT_PATH):
        st.error(f"Ошибка PDF: Файл шрифта не найден: {FONT_PATH}")
        return None
    try:
        pdf.add_font('DejaVu', '', FONT_PATH, uni=True)
        pdf.add_font('DejaVu', 'B', FONT_PATH, uni=True) # Жирный
        pdf.add_font('DejaVu', 'I', FONT_PATH, uni=True) # Курсив
        pdf.font_added = True # Шрифт успешно добавлен
    except Exception as e:
        st.error(f"Ошибка PDF при добавлении шрифта: {e}")
        return None

    pdf.set_auto_page_break(auto=True, margin=15) # Автоматический перенос страниц с отступом снизу 1.5 см
    pdf.alias_nb_pages() # Включаем нумерацию страниц {nb}
    pdf.add_page()

    # --- 1. Исходное изображение ---
    pdf.add_image_from_buffer(original_image_buffer, "1. Исходное изображение")

    # --- 2. Результаты анализа ---
    pdf.chapter_title("2. Результаты анализа")
    analysis_data = {
        "Итоговая оценка состояния": final_assessment,
        "Детали оценки": assessment_details,
        "Рассчитанный индекс": selected_index,
        f"Среднее значение индекса {selected_index} (исходное)": f"{mean_value_original:.3f}" if not math.isnan(mean_value_original) else "N/A",
    }
    pdf.chapter_body(content_dict=analysis_data)

    # --- 3. Рекомендации ---
    pdf.chapter_title("3. Рекомендации")
    pdf.chapter_body(text_list=recommendations)

    # --- 4. Карта индекса ---
    pdf.add_image_from_buffer(index_map_buffer, f"4. Карта индекса {selected_index} (Нормализованная)")

    # --- 5. Расчет площади ---
    # Проверяем, не нужно ли перенести на новую страницу
    if pdf.get_y() > pdf.h - 80: # Если осталось < 80 мм до низа страницы
        pdf.add_page()

    pdf.chapter_title("5. Расчет площади угодий/полей")
    area_data = {
        "Высота полета": f"{flight_params['altitude']:.1f} м",
        "Фокусное расстояние": f"{flight_params['focal_length']:.1f} мм",
        "Ширина сенсора": f"{flight_params['sensor_width']:.1f} мм",
        "Высота сенсора": f"{flight_params['sensor_height']:.1f} мм",
        "--- Результаты ---": "", # Просто разделитель
        "Общая площадь": f"{area_ha:.3f} га ({area_km2:.5f} км²)" if area_ha > 0 else "Не рассчитана",
        "Ширина на земле": f"{ground_width:.2f} м" if ground_width > 0 else "N/A",
        "Высота на земле": f"{ground_height:.2f} м" if ground_height > 0 else "N/A",
    }
    pdf.chapter_body(content_dict=area_data)

    # --- 6. Анализ засушливых участков ---
    if pdf.get_y() > pdf.h - 60:
        pdf.add_page()

    pdf.chapter_title("6. Анализ засушливых участков")
    drought_data = {
       f"Порог нормализованного индекса {selected_index} для определения засухи": f"< {dryness_threshold_normalized:.2f}",
       "Расчетная площадь засушливых участков": f"{dryness_area_ha:.3f} га" if area_ha > 0 else "N/A (общая площадь не рассчитана)",
       "Процент засушливых участков от общей площади": f"{dryness_percent:.1f}%" if area_ha > 0 else "N/A",
       "--- Расчет добавок (Пример) ---": "",
       "Информация": dryness_additive_calc # Текст с расчетом или сообщением
    }
    pdf.chapter_body(content_dict=drought_data)

    # --- Вывод PDF ---
    try:
        # Возвращаем байты PDF для кнопки скачивания
        pdf_bytes = pdf.output() # 'S' возвращает строку, кодируем в latin-1 для байтов
        return bytes(pdf_bytes) # Или return bytes(pdf_bytes) - используйте то имя переменной, которое у вас там
    except Exception as e:
        st.error(f"Ошибка при финальной генерации PDF: {e}")
        return None

# --- Основная функция приложения ---
def main():
    # --- Боковая панель и Навигация ---
    with st.sidebar:
        st.image("https://i.ibb.co/yWXkh9Q/Screenshot-from-2024-05-14-16-04-34.png", width=100) # Пример логотипа, замените URL
        st.title("Меню")

        selected = option_menu(
            menu_title=None, #"Навигация", # None чтобы убрать заголовок
            options=["Анализ", "Карта", "О приложении", "Контакты"],
            icons=["images", "map", "info-circle", "envelope"], # Иконки Bootstrap
            menu_icon="list", # Иконка меню
            default_index=0,
            styles={ # Немного стилизации для меню
                "container": {"padding": "5px !important", "background-color": "#fafafa"},
                "icon": {"color": "#4CAF50", "font-size": "23px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#e8f5e9"},
                "nav-link-selected": {"background-color": "#4CAF50"},
            }
        )
        st.sidebar.markdown("---")
        st.sidebar.info("© 2024-2025 Система Оценки Угодий")

    # --- Отображение выбранной страницы ---

    if selected == "Анализ":
        st.markdown("<h1>🌿 Система оценки состояния угодий 🌿</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Загрузите снимок, выберите индекс и получите анализ</h3>", unsafe_allow_html=True)

        # --- Зона загрузки файла ---
        st.subheader("1. Загрузка изображения")
        st.write("Поддерживаемые форматы: TIF, TIFF, JPG, JPEG, PNG (рекомендуется TIF/TIFF для мультиспектральных данных)")
        uploaded_file = st.file_uploader("Выберите файл...", type=["tif", "tiff", "jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                image_pil_original = Image.open(uploaded_file)
                # Корректируем ориентацию изображения по EXIF
                image_pil_corrected = correct_image_orientation(image_pil_original)
                image_array = np.array(image_pil_corrected) # Работаем с массивом numpy
            except Exception as e:
                st.error(f"Не удалось открыть или обработать изображение: {e}")
                st.stop() # Прекращаем выполнение, если файл не читается

            st.success(f"Файл '{uploaded_file.name}' успешно загружен.")

            if len(image_array.shape) < 3 or image_array.shape[2] < 3:
                 st.error("Ошибка: Изображение должно иметь как минимум 3 канала (RGB).")
                 st.stop()

            num_channels = image_array.shape[2]
            st.write(f"Обнаружено каналов в изображении: **{num_channels}**")

            # --- Определение доступных индексов и порядка каналов ---
            global red_index, green_index, blue_index, nir_index # Указываем, что меняем глобальные переменные
            red_index, green_index, blue_index, nir_index = 0, 1, 2, None # Порядок по умолчанию R, G, B

            # Здесь можно добавить логику для определения порядка каналов, если он не стандартный
            # Например, на основе метаданных или выбора пользователя
            # st.info("Предполагается стандартный порядок каналов: 0-Red, 1-Green, 2-Blue, 3-NIR (если есть).")

            if num_channels >= 4:
                nir_index = 3 # Предполагаем, что NIR - 4й канал (индекс 3)
                st.success("Обнаружен 4-й канал, предполагается NIR. Доступны все индексы.")
                # Список индексов, требующих NIR + RGB индексы
                available_indices = ['NDVI', 'GNDVI', 'ENDVI', 'CVI', 'OSAVI', 'VARI', 'ExG', 'GLI', 'RGBVI']
            else:
                st.warning("Обнаружено только 3 канала (RGB). NIR-зависимые индексы (NDVI, GNDVI и т.д.) не будут рассчитаны.")
                # Только RGB индексы
                available_indices = ['VARI', 'ExG', 'GLI', 'RGBVI']

            # Конвертируем в RGB для отображения и для модели (если она обучена на RGB)
            try:
                image_pil_rgb = image_pil_corrected.convert('RGB')
            except Exception as e:
                st.error(f"Не удалось конвертировать изображение в RGB для отображения: {e}")
                st.stop()

            # Отображение загруженного изображения
            st.image(image_pil_rgb, caption="Загруженное изображение (RGB вид)", use_column_width=True)

            st.markdown("---")
            st.subheader("2. Выбор индекса и параметры анализа")

            col_index, col_threshold = st.columns(2)
            with col_index:
                selected_index = st.selectbox("Выберите вегетационный индекс:", available_indices)
            with col_threshold:
                 # Порог засухи теперь определяется здесь, до вкладок
                 threshold_value = st.slider("Порог засухи (норм. индекс < порога = суше)", 0.0, 1.0, 0.35, step=0.05, key="threshold_main", help="Используется для расчета площади засушливых участков. Меньшие значения нормализованного индекса обычно соответствуют стрессу или отсутствию растительности.")

            st.markdown("---")

            # --- Запуск анализа ---
            if st.button("🚀 Запустить анализ"):
                with st.spinner("Выполняется анализ... Пожалуйста, подождите."):

                    # 1. Загрузка модели
                    model = load_model()
                    if model is None:
                        st.error("Анализ не может быть выполнен без модели.")
                        st.stop()

                    # 2. Предсказание состояния (на RGB версии)
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)), # Размер, на котором обучалась модель
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Если модель обучалась с нормализацией
                    ])
                    prediction_raw = predict_image(image_pil_rgb, model, transform)

                    # 3. Расчёт выбранного индекса
                    index_result_original = calculate_vegetation_index(image_array, selected_index)

                    if index_result_original is None or np.all(np.isnan(index_result_original)):
                        st.error(f"Не удалось рассчитать индекс {selected_index}. Возможно, требуемый канал (например, NIR) отсутствует или произошла ошибка.")
                        st.stop()

                    # 4. Обработка результатов индекса
                    mean_value_original = np.nanmean(index_result_original) # Среднее до нормализации

                    # Нормализация индекса для карты (0-1) и для корректировки предсказания
                    min_val = np.nanmin(index_result_original)
                    max_val = np.nanmax(index_result_original)
                    if max_val - min_val > 1e-7:
                         normalized_index = (index_result_original - min_val) / (max_val - min_val)
                    else:
                         # Если все значения почти одинаковы, карта будет одного цвета
                         normalized_index = np.full_like(index_result_original, 0.5) # Заполняем средним значением (0.5)
                         st.info("Значения индекса по всему изображению практически одинаковы.")

                    mean_normalized_value = np.nanmean(normalized_index) # Среднее нормализованное значение

                    # --- Корректировка предсказания модели на основе индекса ---
                    final_assessment = prediction_raw # Начинаем с вывода модели
                    assessment_note = "" # Дополнительное примечание
                    # Порог для "здорового" нормализованного индекса (0.0 - 1.0)
                    # !!! НАСТРОЙТЕ ЭТОТ ПОРОГ ПРИ НЕОБХОДИМОСТИ !!!
                    HEALTHY_INDEX_THRESHOLD = 0.5

                    if "засуха" in prediction_raw.lower() and mean_normalized_value > HEALTHY_INDEX_THRESHOLD:
                        assessment_note = (f"Примечание: Модель предсказала засуху/стресс, но среднее значение "
                                           f"нормализованного индекса ({mean_normalized_value:.3f}) выше порога "
                                           f"({HEALTHY_INDEX_THRESHOLD}), что может указывать на более здоровое состояние растительности.")
                        final_assessment += " | Индекс: Оптимально" # Добавляем информацию к выводу
                        st.info(assessment_note)
                    elif ("оптимально" in prediction_raw.lower() or "хорошие условия" in prediction_raw.lower()) and mean_normalized_value < (HEALTHY_INDEX_THRESHOLD - 0.15): # Порог чуть ниже
                         assessment_note = (f"Примечание: Модель предсказала оптимальное состояние, но среднее значение "
                                            f"нормализованного индекса ({mean_normalized_value:.3f}) ниже ожидаемого "
                                            f"(<{HEALTHY_INDEX_THRESHOLD - 0.15}), что может указывать на некоторый стресс у растений или редкую растительность.")
                         final_assessment += " | Индекс: Возможен стресс"
                         st.warning(assessment_note)
                    # --- Конец корректировки ---

                    # 5. Формирование Рекомендаций (основаны на исходном предсказании модели)
                    current_recommendations = []
                    if "засуха" in prediction_raw.lower(): # Используем исходное предсказание для рекомендаций
                        rec1 = "Рекомендуется проверить влажность почвы и при необходимости провести полив."
                        rec2 = "Рассмотреть возможность внесения водоудерживающих добавок для улучшения удержания влаги."
                        rec3 = "Проанализировать карту индекса для выявления наиболее проблемных зон."
                        current_recommendations.extend([rec1, rec2, rec3])
                    elif "оптимально" in prediction_raw.lower():
                        rec1 = "Поддерживать текущий режим ухода и мониторинга."
                        rec2 = "Рассмотреть плановое внесение удобрений для поддержания плодородия."
                        rec3 = "Продолжать регулярный мониторинг состояния угодий с помощью индексов."
                        current_recommendations.extend([rec1, rec2, rec3])
                    else: # Если предсказание неопределенное
                        current_recommendations.append("Проведите дополнительный осмотр угодий для уточнения состояния.")

                    st.markdown("---")
                    st.subheader("3. Результаты Анализа")

                    # Отображение итоговой оценки и рекомендаций
                    st.metric(label="Итоговая оценка состояния", value=final_assessment)
                    if assessment_note:
                         st.caption(assessment_note) # Показываем примечание, если оно есть
                    st.write(f"Среднее значение индекса **{selected_index}** (ненорм.): **{mean_value_original:.3f}**")
                    st.write(f"Среднее значение индекса **{selected_index}** (норм.): **{mean_normalized_value:.3f}**")

                    st.markdown("##### Рекомендации:")
                    for rec in current_recommendations:
                        st.markdown(f"- {rec}")

                    # --- Вкладки с деталями ---
                    st.markdown("---")
                    st.subheader("4. Детальные результаты")
                    tab1, tab2, tab3 = st.tabs([
                        f"🗺️ Карта индекса {selected_index}",
                        "📏 Расчет площади",
                        f"💧 Площадь засухи (Индекс < {threshold_value:.2f})"
                    ])

                    # --- Переменные для расчета площади и засухи ---
                    gsd_width_m_px, gsd_height_m_px = None, None
                    area_m2, area_ha, area_km2 = 0, 0, 0
                    ground_width_m, ground_height_m = 0, 0
                    flight_params_dict = {}
                    index_map_buffer = None # Буфер для PDF
                    dryness_area_ha = 0
                    dryness_percent = 0
                    dryness_additive_calculation_text = "Расчет не производился." # Текст для PDF

                    # --- Вкладка 1: Карта индекса ---
                    with tab1:
                        st.subheader(f"Карта индекса {selected_index} (Нормализованная)")
                        try:
                            index_map_fig, ax = plt.subplots(figsize=(10, 8)) # Размер фигуры
                            # Используем нормализованные значения (0-1) для RdYlGn
                            im = ax.imshow(normalized_index, cmap='RdYlGn', vmin=0, vmax=1)
                            ax.set_title(f"Нормализованный индекс {selected_index}")
                            ax.axis('off')
                            # Colorbar с оригинальными значениями
                            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
                            cbar.set_label(f"Значение {selected_index}", rotation=270, labelpad=15)
                            # Устанавливаем тики на colorbar на основе оригинальных min/max
                            ticks = np.linspace(0, 1, 6)
                            tick_labels = [f"{min_val + (max_val-min_val)*v:.2f}" for v in ticks]
                            cbar.set_ticks(ticks)
                            cbar.set_ticklabels(tick_labels)

                            st.pyplot(index_map_fig)

                            # Сохраняем карту в буфер для PDF
                            index_map_buffer = io.BytesIO()
                            index_map_fig.savefig(index_map_buffer, format='png', bbox_inches='tight', dpi=150)
                            index_map_buffer.seek(0)
                            plt.close(index_map_fig) # Закрываем фигуру
                        except Exception as e:
                            st.error(f"Ошибка при построении карты индекса: {e}")
                            index_map_buffer = None # Сбрасываем буфер, если ошибка

                    # --- Вкладка 2: Расчет площади ---
                    with tab2:
                        st.subheader("Расчет площади угодий по параметрам съемки")
                        image_width_px, image_height_px = image_pil_corrected.size
                        metadata = get_image_metadata(image_pil_corrected)

                        # Попытка получить фокусное расстояние из EXIF
                        focal_length_exif = metadata.get('FocalLength', None)
                        focal_length_default = 35.0 # Значение по умолчанию
                        if focal_length_exif is not None:
                            try:
                                if isinstance(focal_length_exif, tuple) and len(focal_length_exif) == 2 and focal_length_exif[1] != 0:
                                    focal_length_default = float(focal_length_exif[0]) / float(focal_length_exif[1])
                                else:
                                    focal_length_default = float(focal_length_exif)
                                st.caption(f"Фокусное расстояние из метаданных: {focal_length_default:.2f} мм (используется по умолчанию)")
                            except (ValueError, TypeError, ZeroDivisionError):
                                st.caption("Не удалось прочитать фокусное расстояние из EXIF. Используется значение по умолчанию (35.0 мм).")
                                focal_length_default = 35.0
                        else:
                            st.caption("Фокусное расстояние в метаданных не найдено. Используется значение по умолчанию (35.0 мм).")

                        st.markdown("##### Введите параметры полета и камеры:")
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            flight_altitude = st.number_input("Высота полета (м)", min_value=1.0, value=100.0, step=5.0, key="altitude_input", help="Средняя высота над уровнем земли.")
                            sensor_width = st.number_input("Ширина сенсора (мм)", min_value=1.0, value=23.5, step=0.1, key="sensor_w_input", help="Например, APS-C ~23.5мм, Full Frame ~36мм.")
                        with f_col2:
                            focal_length = st.number_input("Фокусное расстояние (мм)", min_value=1.0, value=focal_length_default, step=1.0, key="focal_input")
                            sensor_height = st.number_input("Высота сенсора (мм)", min_value=1.0, value=15.6, step=0.1, key="sensor_h_input", help="Например, APS-C ~15.6мм, Full Frame ~24мм.")

                        # Сохраняем параметры для отчета
                        flight_params_dict = {
                            "altitude": flight_altitude, "focal_length": focal_length,
                            "sensor_width": sensor_width, "sensor_height": sensor_height
                        }

                        # Расчет GSD и площади
                        try:
                            gsd_width_m_px, gsd_height_m_px = calculate_gsd(
                                flight_altitude, focal_length, sensor_width, sensor_height, image_width_px, image_height_px
                            )
                            if gsd_width_m_px > 0 and gsd_height_m_px > 0:
                                area_m2, ground_width_m, ground_height_m = calculate_image_area(
                                    gsd_width_m_px, gsd_height_m_px, image_width_px, image_height_px
                                )
                                area_ha = area_m2 / 10000
                                area_km2 = area_m2 / 1_000_000

                                st.markdown("##### Результаты расчета площади:")
                                st.metric("Разрешение на местности (GSD)", f"{gsd_width_m_px*100:.2f} см/пикс (ширина), {gsd_height_m_px*100:.2f} см/пикс (высота)")
                                m_col1, m_col2, m_col3 = st.columns(3)
                                m_col1.metric("Площадь покрытия", f"{area_ha:.3f} га")
                                m_col2.metric("Ширина на земле", f"{ground_width_m:.2f} м")
                                m_col3.metric("Высота на земле", f"{ground_height_m:.2f} м")
                                st.caption(f"Площадь покрытия: {area_km2:.5f} км²")
                            else:
                                st.warning("Не удалось рассчитать GSD. Проверьте параметры полета и камеры.")
                                area_ha = 0 # Сбрасываем площадь, если расчет не удался
                        except Exception as e:
                            st.error(f"Ошибка при расчете площади: {e}")
                            area_ha = 0

                    # --- Вкладка 3: Площадь засухи ---
                    with tab3:
                        st.subheader(f"Расчет площади участков с индексом {selected_index} < {threshold_value:.2f}")

                        if area_ha > 0 and gsd_width_m_px is not None and gsd_height_m_px is not None:
                            # Используем ту же нормализованную карту
                            dryness_mask = normalized_index < threshold_value
                            dryness_pixels = np.sum(dryness_mask) # Сумма пикселей в маске

                            # Расчет площади одного пикселя (среднее GSD)
                            pixel_area_m2 = gsd_width_m_px * gsd_height_m_px
                            dryness_area_m2 = dryness_pixels * pixel_area_m2
                            dryness_area_ha = dryness_area_m2 / 10000
                            dryness_percent = (dryness_area_ha / area_ha) * 100 if area_ha > 0 else 0

                            st.metric("Площадь засушливых участков", f"{dryness_area_ha:.3f} га")
                            st.metric("Процент от общей площади", f"{dryness_percent:.1f}%")

                            # Пример расчета добавок
                            if dryness_area_ha > 0 and "засуха" in prediction_raw.lower(): # Используем исходное предсказание
                                recommended_dose_t_per_ha = 4.64 # Пример дозы в тоннах на гектар
                                total_additive = dryness_area_ha * recommended_dose_t_per_ha
                                st.markdown("##### Расчет водоудерживающих добавок (Пример):")
                                st.write(f"- Рекомендуемая доза: {recommended_dose_t_per_ha:.2f} т/га (участка с индексом < {threshold_value:.2f})")
                                st.write(f"- **Общее количество добавки:** **{total_additive:.2f} т** (на {dryness_area_ha:.3f} га)")
                                dryness_additive_calculation_text = (f"Рекомендуемая доза водоудерживающих добавок: {recommended_dose_t_per_ha:.2f} т/га. "
                                                                   f"Требуемое количество на {dryness_area_ha:.3f} га: {total_additive:.2f} т.")
                            elif "засуха" not in prediction_raw.lower():
                                 st.info("Состояние оценено как оптимальное, расчет водоудерживающих добавок не требуется.")
                                 dryness_additive_calculation_text = "Состояние оценено как оптимальное, расчет водоудерживающих добавок не требуется."
                            else: # Если площадь засухи = 0
                                 st.info("Не обнаружено участков с индексом ниже заданного порога.")
                                 dryness_additive_calculation_text = "Не обнаружено участков с индексом ниже заданного порога."

                        else:
                            st.warning("Расчет площади засухи невозможен. Сначала необходимо успешно рассчитать общую площадь во вкладке 'Расчет площади'.")
                            dryness_additive_calculation_text = "Расчет невозможен (общая площадь не рассчитана)."

                    # --- Кнопка генерации PDF (после всех вкладок) ---
                    st.markdown("---")
                    st.subheader("5. Генерация PDF отчета")

                    # Сохраняем исходное изображение в буфер для PDF
                    original_image_buffer = io.BytesIO()
                    try:
                         image_pil_rgb.save(original_image_buffer, format='PNG')
                         original_image_buffer.seek(0)
                    except Exception as e:
                         st.error(f"Ошибка сохранения исходного изображения для PDF: {e}")
                         original_image_buffer = None

                    # Проверяем наличие всех данных для PDF
                    if original_image_buffer and index_map_buffer and flight_params_dict:
                        pdf_bytes = generate_pdf_report(
                             original_image_buffer=original_image_buffer,
                             final_assessment=final_assessment,
                             assessment_details=assessment_note if assessment_note else f"Среднее норм. значение {selected_index}: {mean_normalized_value:.3f}",
                             selected_index=selected_index,
                             mean_value_original=mean_value_original,
                             index_map_buffer=index_map_buffer,
                             recommendations=current_recommendations,
                             area_ha=area_ha, ground_width=ground_width_m, ground_height=ground_height_m, area_km2=area_km2,
                             flight_params=flight_params_dict,
                             dryness_threshold_normalized=threshold_value,
                             dryness_area_ha=dryness_area_ha, dryness_percent=dryness_percent,
                             dryness_additive_calc=dryness_additive_calculation_text
                        )

                        if pdf_bytes:
                             st.download_button(
                                 label="📥 Скачать PDF отчет",
                                 data=pdf_bytes,
                                 file_name=f"report_{selected_index}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                 mime="application/pdf",
                                 help="Скачать полный отчет с результатами анализа в формате PDF."
                             )
                        else:
                             st.error("Не удалось сгенерировать PDF файл отчета.")
                    else:
                         st.warning("Не хватает данных для генерации PDF отчета (возможно, не удалось создать карту индекса или рассчитать площадь).")

            # Если кнопка "Запустить анализ" не нажата, ничего не происходит после загрузки

        # Если файл не загружен
        else:
            st.info("Пожалуйста, загрузите изображение для начала анализа.")

    elif selected == "Карта":
        st.markdown("<h1>🗺️ Интерактивная карта</h1>", unsafe_allow_html=True)
        st.markdown("### Выберите интересующий регион", unsafe_allow_html=True)
        interactive_map() # Вызываем функцию отображения карты
        st.caption("Эта карта предназначена для обзора. В текущей версии координаты с карты не используются для расчетов.")

    elif selected == "О приложении":
        st.markdown("<h1>📄 О приложении</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            """
            **Система оценки состояния угодий (Версия 1.1)** - это инструмент для агрономов, фермеров и исследователей,
            предназначенный для анализа состояния растительности на основе мультиспектральных или RGB изображений.

            **Основные возможности:**
            * **Загрузка изображений:** Поддержка форматов TIF, TIFF, JPG, JPEG, PNG.
            * **Расчет вегетационных индексов:** Вычисление стандартных индексов, таких как NDVI (при наличии NIR), VARI, ExG, GLI и других.
            * **Оценка состояния:** Использование простой модели ИИ для предварительной классификации состояния (засуха/оптимально), с корректировкой на основе рассчитанного индекса.
            * **Визуализация:** Построение наглядной карты распределения выбранного индекса по полю.
            * **Рекомендации:** Предоставление базовых рекомендаций по уходу в зависимости от оценки состояния.
            * **Расчет площади:** Определение примерной площади покрытия снимка и GSD на основе параметров полета и камеры.
            * **Анализ зон стресса:** Расчет площади участков, где значение индекса ниже заданного порога.
            * **Генерация отчетов:** Создание и скачивание подробного PDF-отчета со всеми результатами анализа.

            **Принцип работы:**
            1.  Пользователь загружает изображение угодий.
            2.  Приложение определяет количество каналов и доступные индексы.
            3.  Пользователь выбирает индекс для расчета и задает параметры (например, порог засухи).
            4.  Приложение рассчитывает индекс, выполняет предсказание с помощью ИИ, корректирует его и формирует рекомендации.
            5.  Рассчитываются площади (общая и зон стресса) на основе введенных параметров съемки.
            6.  Результаты отображаются на экране и могут быть сохранены в виде PDF-отчета.

            **Ограничения:**
            * Точность оценки ИИ зависит от качества и репрезентативности данных, на которых обучалась модель (`model.pth`). Текущая модель является примером.
            * Точность расчета площади зависит от точности введенных параметров полета и камеры, а также от рельефа местности (предполагается плоская поверхность).
            * Определение порядка каналов (R, G, B, NIR) основано на стандартных предположениях. Для нестандартных камер может потребоваться адаптация кода.
            """
        )

    elif selected == "Контакты":
        st.markdown("<h1>📧 Контакты</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.write(
            """
            По вопросам, связанным с работой приложения, предложениям по улучшению или сообщениям об ошибках,
            пожалуйста, обращайтесь:

            **Разработчик:** (Ваше Имя или Название Компании)

            * **Email:** `Alina@gmail.com` 
            * **Телефон:** `+X (XXX) XXX-XX-XX` 
            * **Веб-сайт/GitHub:** 

            Мы ценим ваши отзывы!
            """
        )

if __name__ == "__main__":
    # --- Проверка наличия файла шрифта перед запуском ---
    if not os.path.exists(FONT_PATH):
        st.warning(
            f"ВНИМАНИЕ: Файл шрифта '{FONT_PATH}' не найден! "
            f"Кириллица в PDF-отчете может отображаться некорректно. "
            f"Пожалуйста, скачайте TTF-файл шрифта (например, DejaVuSans.ttf) и поместите его в ту же папку, что и скрипт."
        )
    main()