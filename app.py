from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import os
from model import predict_emotions
from collections import Counter

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"])

# Diretório para salvar temporariamente os frames
OUTPUT_DIR = 'video_frames'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_output_dir():
    """Remove todos os arquivos em OUTPUT_DIR."""
    for file in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Falha ao deletar {file_path}. Motivo: {e}')

@app.route('/')
def index():
    # Renderiza o index.html ao acessar a rota principal
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Receber o vídeo e o intervalo
    video_file = request.files.get('video')
    interval = int(request.form.get('interval', 10))  # Padrão para 10 segundos, se não especificado

    if not video_file:
        return jsonify({"error": "Nenhum vídeo enviado"}), 400

    # Salvar o vídeo temporariamente
    video_path = os.path.join(OUTPUT_DIR, video_file.filename)
    video_file.save(video_path)

    # Carregar o vídeo com OpenCV
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))  # Taxa de quadros por segundo do vídeo
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Duração total em segundos

    # Calcula o número de segmentos
    num_segments = int(duration // interval)
    segments_emotions = []  # Armazena a emoção predominante de cada segmento
    all_frames_emotions = []  # Armazena a emoção de cada frame

    for segment in range(num_segments):
        # Definir o ponto de início do segmento
        start_frame = segment * interval * fps
        end_frame = min((segment + 1) * interval * fps, total_frames)
        frames = []

        # Captura todos os frames dentro do segmento atual
        for frame_number in range(int(start_frame), int(end_frame), fps):  # Captura um frame por segundo no intervalo
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if ret:
                frames.append(frame)
            else:
                break

        # Analisar emoções dos frames capturados no segmento atual
        segment_emotions = predict_emotions(frames)
        
        # Adicionar emoções de cada frame ao array `all_frames_emotions`
        for i, emotion in enumerate(segment_emotions):
            all_frames_emotions.append({
                "segmento": segment + 1,
                "frame": start_frame + i * fps,
                "emocao": emotion
            })

        # Determinar a emoção predominante no segmento atual
        emotion_counts = Counter(segment_emotions)
        predominant_emotion = emotion_counts.most_common(1)[0][0]
        segments_emotions.append({
            "segmento": segment + 1,
            "emocao_predominante": predominant_emotion
        })

    video.release()
    os.remove(video_path)  # Remover o vídeo temporário

    # Retorna o JSON com as emoções por frame e as emoções predominantes em cada segmento
    return jsonify({
        "message": "Processamento concluído",
        "frames": all_frames_emotions,
        "segments": segments_emotions
    }), 200

@app.route('/analyze_emotions', methods=['GET'])
def analyze_emotions():
    # Usa a função `predict_emotions` para analisar as emoções nas imagens processadas
    emotions = predict_emotions(OUTPUT_DIR)
    return jsonify({"emotions": emotions}), 200

if __name__ == "__main__":
    app.run()
