from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import os
from model import predict_emotions

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
    # Limpar frames do vídeo anterior
    clear_output_dir()

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

    # Calcula o número de pedaços
    num_segments = int(duration // interval)
    captured_frames = []

    for segment in range(num_segments):
        # Definir o ponto de início do segmento
        start_frame = segment * interval * fps

        # Capturar 1 frame por segundo no segmento
        for sec in range(interval):
            frame_number = start_frame + sec * fps
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if ret:
                # Salvar o frame como imagem
                frame_filename = f"frame_segment_{segment}_second_{sec}.jpg"
                frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                cv2.imwrite(frame_path, frame)
                captured_frames.append(frame_filename)
            else:
                break

    video.release()
    os.remove(video_path)  # Remover o vídeo 0temporário

    return jsonify({"message": "Processamento concluído", "frames": captured_frames}), 200

@app.route('/analyze_emotions', methods=['GET'])
def analyze_emotions():
    # Usa a função `predict_emotions` para analisar as emoções nas imagens processadas
    emotions = predict_emotions(OUTPUT_DIR)
    return jsonify({"emotions": emotions}), 200

if __name__ == '__main__':
    app.run(debug=True)
