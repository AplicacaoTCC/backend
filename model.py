import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.externals import joblib
from PIL import Image

# Classes de emoções
class_mapping = ['Tedio', 'Confusao', 'Engajamento', 'Frustracao']

# Carregar o modelo de emoções e o classificador SVM
model = load_model('FER_Model.hdf5')
clf = joblib.load('classifier_svm.pkl')

# Carregar o classificador de rosto Haar Cascade
cascade_path = os.path.join('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Função para processar imagem e prever emoções com probabilidades
def get_emotion_classification(dest):
    frame = cv2.imread(dest)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Cor azul (BGR) e espessura 2
        img = gray[y:y+h, x:x+w]
        clahe_image = clahe.apply(img)
        img1 = Image.fromarray(clahe_image).resize((48, 48))
        img1 = np.asarray(img1).reshape((48, 48, 1))
        img1 = np.expand_dims(img1, axis=0)

        # Extrair features com o modelo
        f = model.predict(img1)
        reshaped_image = np.pad(f, (0, 3200 - f.size), 'constant').reshape(1, 3200) if f.size < 3200 else f[:3200].reshape(1, 3200)
        probabilities = clf.predict_proba(reshaped_image)[0]  # Retorna uma lista de probabilidades para cada classe
        
        emotion_probabilities = {class_mapping[i]: prob * 100 for i, prob in enumerate(probabilities)}
        
        return emotion_probabilities, frame

    # Caso nenhum rosto seja detectado
    zeroed_emotions = {emotion: 0.0 for emotion in class_mapping}
    return zeroed_emotions, frame
