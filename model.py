# model.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo .h5
model = load_model("fer_model_best.h5")

# Carregar o classificador de rosto Haar Cascade
cascade_path = os.path.join('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

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
def load_and_preprocess_faces(frames, target_size=(48, 48)):
    images = []
    for frame in frames:
        # Converter o frame para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar o rosto no frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Se pelo menos um rosto for detectado, pegar o primeiro rosto
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Coordenadas do rosto
            face = gray_frame[y:y + h, x:x + w]  # Recortar a região do rosto

            # Redimensionar o rosto para o tamanho desejado (48x48)
            face = cv2.resize(face, target_size)

            # Normalizar a imagem
            face = face / 255.0

            # Expandir as dimensões para (48, 48, 1) para que o modelo as interprete corretamente
            face = np.expand_dims(face, axis=-1)

            # Adicionar o rosto preprocessado à lista de imagens
            images.append(face)

    # Converter a lista em um array NumPy para ser usado na predição
    return np.array(images)


def predict_emotions(frames):
    # Carregar e preprocessar os rostos dos frames
    test_images = load_and_preprocess_faces(frames)

    # Verificar se há rostos detectados
    if test_images.size == 0:
        return ["Nenhum rosto detectado"] * len(frames)

    # Realizar predições com o modelo
    label_ps = model.predict(test_images)

    # Armazenar resultados de emoções
    emotions = []

    for prediction in label_ps:
        # Encontrar o índice da maior probabilidade
        predicted_label = np.argmax(prediction)
        # Mapear para o nome da emoção
        emotion = class_mapping[predicted_label]
        # Adicionar a emoção prevista ao resultado
        emotions.append(emotion)

    return emotions


