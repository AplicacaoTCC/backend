# model.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo .h5
model = load_model("fer_model_best.h5")

# Carregar o classificador de rosto Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapeamento de classes
class_mapping = {
    0: 'Raiva',
    1: 'Desgosto',
    2: 'Medo',
    3: 'Felicidade',
    4: 'Tristeza',
    5: 'Surpresa',
    6: 'Neutro'
}


# Função para carregar, detectar o rosto e preprocessar as imagens
def load_and_preprocess_faces(image_dir, target_size=(48, 48)):
    images = []
    file_names = []  # Para armazenar nomes de arquivos para identificação de saída
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)

        # Carregar a imagem em escala de cinza
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue  # Ignora arquivos que não são imagens

        # Detectar o rosto na imagem
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Se pelo menos um rosto for detectado, pegar o primeiro rosto
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Coordenadas do rosto
            face = image[y:y + h, x:x + w]  # Recortar a região do rosto

            # Redimensionar o rosto para o tamanho desejado (48x48)
            face = cv2.resize(face, target_size)

            # Normalizar a imagem
            face = face / 255.0

            # Expandir as dimensões para (48, 48, 1) para que o modelo as interprete corretamente
            face = np.expand_dims(face, axis=-1)

            # Adicionar o rosto preprocessado à lista de imagens
            images.append(face)
            file_names.append(file_name)  # Armazena o nome da imagem atual

    # Converter a lista em um array NumPy para ser usado na predição
    return np.array(images), file_names


# Função para prever emoções nas imagens
def predict_emotions(image_dir):
    # Carregar e preprocessar os rostos
    test_images, file_names = load_and_preprocess_faces(image_dir)

    # Realizar predições com o modelo
    label_ps = model.predict(test_images)

    # Armazenar resultados de emoções
    emotions = []

    for i, prediction in enumerate(label_ps):
        # Encontrar o índice da maior probabilidade
        predicted_label = np.argmax(prediction)
        # Mapear para o nome da emoção
        emotion = class_mapping[predicted_label]
        # Armazenar o nome do arquivo e a emoção prevista
        emotions.append({"imagem": file_names[i], "emocao": emotion})

    return emotions
