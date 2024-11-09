from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import os
from model import get_emotion_classification

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
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Limpar frames do vídeo anterior
    clear_output_dir()

    # Receber o vídeo
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "Nenhum vídeo enviado"}), 400

    # Salvar o vídeo temporariamente
    video_path = os.path.join(OUTPUT_DIR, video_file.filename)
    video_file.save(video_path)

    # Carregar o vídeo com OpenCV
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))  # Taxa de quadros por segundo do vídeo
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    emotion_results = []  # Armazena os resultados de emoções

    # Processar um frame por segundo
    for frame_number in range(0, total_frames, fps):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if ret:
            # Salvar o frame temporariamente
            frame_path = os.path.join(OUTPUT_DIR, f"temp_frame.jpg")
            cv2.imwrite(frame_path, frame)

            # Calcular o tempo em minutos e segundos
            time_in_seconds = frame_number / fps
            minutes = int(time_in_seconds // 60)
            seconds = int(time_in_seconds % 60)
            time_formatted = f"{minutes:02}:{seconds:02}"

            # Analisar a emoção do frame atual
            emotions = get_emotion_classification(frame_path)
            
            if emotions:
                emotion_results.append({
                    "time": time_formatted,
                    "emotions": emotions  # Probabilidades das emoções para o frame
                })
        else:
            break

    video.release()
    os.remove(video_path)  # Remover o vídeo temporário
    return jsonify({"message": "Processamento concluído", "results": emotion_results}), 200

if __name__ == "__main__":
    app.run()
