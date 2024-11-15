from flask import Flask, make_response, request, render_template, jsonify, send_from_directory, stream_with_context, Response
from flask_cors import CORS
import random, time, json, os, cv2
from model import get_emotion_classification

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200", "https://aplicacao-principal.vercel.app", "https://aplicacaoprincipal.onrender.com"])

# Diretório para salvar temporariamente os frames
OUTPUT_DIR = 'video_frames'
HIGHLIGHTED_DIR = 'highlighted_frames'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)
video_path = None 

def clear_output_dir():
    """Remove todos os arquivos em OUTPUT_DIR e HIGHLIGHTED_DIR."""
    for folder in [OUTPUT_DIR, HIGHLIGHTED_DIR]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Falha ao deletar {file_path}. Motivo: {e}')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST', 'GET'])
def process_video():
    global video_path

    if request.method == 'POST':
        clear_output_dir()

        # Receber o vídeo
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({"error": "Nenhum vídeo enviado"}), 400

        # Salvar o vídeo temporariamente
        video_path = os.path.join(OUTPUT_DIR, video_file.filename)
        video_file.save(video_path)

        return jsonify({"message": "Vídeo recebido com sucesso"}), 200

    elif request.method == 'GET':
        if not video_path:
            return jsonify({"error": "Nenhum vídeo para processar"}), 400

        def generate():
            # Carregar o vídeo
            video = cv2.VideoCapture(video_path)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            highlights_saved = 0

            # Processar um frame por segundo
            for frame_number in range(0, total_frames, fps):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video.read()
                if not ret:
                    break  # Encerrar o loop se não houver mais frames para processar

                # Salvar o frame temporariamente
                frame_path = os.path.join(OUTPUT_DIR, f"temp_frame.jpg")
                cv2.imwrite(frame_path, frame)

                # Calcular o tempo em minutos e segundos
                time_in_seconds = frame_number / fps
                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)
                time_formatted = f"{minutes:02}:{seconds:02}"

                # Analisar a emoção do frame atual
                result = get_emotion_classification(frame_path)
                
                if result is not None:
                    emotions, frame_with_markings = result
                    
                    # Calcula o progresso atual
                    current_frame = (frame_number // fps) + 1  # Contador de frames analisados
                    progress = {
                        "time": time_formatted,
                        "emotions": emotions,
                        "frame": f"{current_frame}/{total_frames // fps}"
                    }
                    
                    # Enviar atualizações incrementais ao front-end
                    yield f"data: {json.dumps(progress)}\n\n"

                    # Salvar alguns frames destacados
                    if random.random() <= 0.1 and highlights_saved < 8:
                        highlight_frame_path = os.path.join(HIGHLIGHTED_DIR, f"{minutes:02}{seconds:02}.png")
                        cv2.imwrite(highlight_frame_path, frame_with_markings)
                        highlights_saved += 1

                time.sleep(0.1)  # Simular tempo de processamento

            # Libera o vídeo e remove o arquivo temporário
            video.release()
            os.remove(video_path)
            
            # Enviar mensagem final para indicar que o processamento está concluído
            yield f"data: {json.dumps({'message': 'Processamento concluído'})}\n\n"

        # Retornar a resposta de stream contínuo com o cabeçalho adequado
        return Response(
            stream_with_context(generate()), 
            content_type='text/event-stream',
            headers={
                'Access-Control-Allow-Origin': '*',  # Permite todas as origens; altere conforme necessário
                'Cache-Control': 'no-cache',         # Recomendado para SSE
                'Connection': 'keep-alive'           # Mantém a conexão ativa
            })
        

@app.route('/highlighted_frames', methods=['GET'])
def get_highlighted_frames():
    """Retorna a lista de frames destacados com URLs."""
    frames = []
    for file_name in os.listdir(HIGHLIGHTED_DIR):
        if file_name.endswith(".png"):
            file_path = os.path.join(HIGHLIGHTED_DIR, file_name)
            frames.append({
                "filename": file_name,
                "url": f"/highlighted_frames/{file_name}"
            })
    return jsonify({"frames": frames}), 200


@app.route('/highlighted_frames/<filename>', methods=['GET'])
def serve_highlighted_frame(filename):
    """Serve um frame de destaque específico."""
    response = make_response(send_from_directory(HIGHLIGHTED_DIR, filename))
    response.headers["Access-Control-Allow-Origin"] = "*"  # Permite todas as origens (para testes)
    return response

if __name__ == "__main__":
    app.run()
