# Importando bibliotecas necessárias
import cv2
import pandas as pd
from datetime import datetime
import os
import numpy as np

# Função para obter máscara de movimento limpa
def get_motion_mask(fg_mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Aplica threshold para obter uma imagem binária
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Aplica mediana para reduzir o ruído
    motion_mask = cv2.medianBlur(thresh, 5)
    
    # Aplica operações morfológicas de abertura e fechamento com kernel menor
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    
    return motion_mask

 # Função para realizar supressão de não-máximos
def non_max_suppression(bboxes, scores, threshold=0.3):

    # Extrair as coordenada x1, x2, y1 e y2 das caixas delimitadoras
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # Calcula a área de cada caixa delimitadora
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Ordena os índices das caixas delimitadoras com base em seus scores de confiança em ordem decrescente (do maior para o menor).
    order = scores.argsort()[::-1]

    # Seleção: Seleciona a caixa com o maior score
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Cálculo das Interseções
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Dimensões da Interseção: Calcula a largura (w) e altura (h) das regiões de interseção.
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Área da Interseção: Calcula a área de interseção
        inter = w * h

        # IoU (Intersection over Union): Calcula o grau de sobreposição (IoU) entre a caixa selecionada e as outras caixas.
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Se o IoU for superior ao limiar threshold, as caixas serão consideradas sobrepostas e potencialmente redundantes
        inds = np.where(ovr <= threshold)[0]

        # Atualização: Atualiza a lista order removendo as caixas suprimidas.
        order = order[inds + 1]

    return keep

# Função principal para processar o vídeo e detectar movimento
def process_video(video_path, skip_frames=5, output_dir='./jupyter_notebooks/motion_detection/02-output'):

    # Variáveis e Inicializações
    motion_list = [None, None]
    contour_area_threshold = 10000
    var_threshold = 50
    use_hist_eq = False

    # Inicializa o subtrator de fundo MOG2 com parâmetros ajustados
    back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=var_threshold, detectShadows=True)

    # Abre o vídeo. Se não for possível, a função exibe um erro e retorna.
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Erro ao abrir o vídeo {video_path}")
        return

    # Extrai o nome do arquivo de vídeo
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Define o caminho completo onde os frames processados e o background serão salvos.
    video_output_dir = os.path.join(output_dir, video_name)

    # Cria o diretório de saída, se ele não existir.
    os.makedirs(video_output_dir, exist_ok=True)

    # Contador de frames processados.
    frame_count = 0

    # Lista para armazenar os frames coloridos para posterior cálculo do background.
    frames = []

    while True:

        # Lê o próximo frame do vídeo. Se não houver mais frames, o loop é interrompido.
        check, frame = video.read()
        if not check:
            break

        frame_count += 1

        # Pula a detecção para alguns frames, se necessário, para reduzir a carga computacional.
        if frame_count % skip_frames != 0:
            continue

        # Converte o frame atual para escala de cinza, o que é necessário para a subtração de fundo.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Se ativado, aplica equalização de histograma para melhorar o contraste na imagem em escala de cinza.
        if use_hist_eq:
            gray = cv2.equalizeHist(gray)

        # Aplica o subtrator de fundo MOG2 para obter a máscara de movimento.
        fg_mask = back_sub.apply(gray)

        # Armazena o frame original (em cores) na lista de frames.
        frames.append(frame)

        # Aplica a função `get_motion_mask` para limpar e processar a máscara de movimento.
        motion_mask = get_motion_mask(fg_mask)

        # Encontra contornos na máscara de movimento
        cnts, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lista de caixas delimitadoras que contêm movimento relevante.
        detections = []
        for contour in cnts:

            # Filtra os contornos com base na área mínima (contour_area_threshold)
            if cv2.contourArea(contour) < contour_area_threshold:
                continue

            # Calcula o retângulo delimitador para cada contorno relevante.
            (x, y, w, h) = cv2.boundingRect(contour)
            detections.append((x, y, x+w, y+h))

        # Aplica supressão de não-máximos
        detections = np.array(detections)
        if len(detections) > 0:

            # Extrai as coordenadas das caixas delimitadoras para aplicar supressão de não-máximos.
            bboxes = detections[:, :4]

            # Scores de confiança (todos iguais a 1, pois os scores reais não são fornecidos).
            scores = np.ones((len(bboxes),))

            # Resultados da supressão de não-máximos (índices das caixas que serão mantidas).
            keep = non_max_suppression(bboxes, scores, threshold=0.3)
            for i in keep:
                (x, y, x2, y2) = bboxes[i]

                # Desenha um retângulo verde ao redor das detecções mantidas.
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)

        # Salvando Frames com Movimento

        # Atualiza a lista para refletir se movimento foi detectado nos últimos dois frames.
        motion_list.append(len(detections) > 0)
        motion_list = motion_list[-2:]

        if len(detections) > 0:
            frame_path = os.path.join(video_output_dir, f'frame_{frame_count}.jpg')

            # Se movimento foi detectado, salva o frame atual em um arquivo JPG.
            cv2.imwrite(frame_path, frame)
            print(f"Frame {frame_count} salvo com movimento detectado.")

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    if frames:
        # Calcula a imagem de background como a mediana de todos os frames capturados.
        background = np.median(frames, axis=0).astype(np.uint8)

        # Define nome e local.
        background_path = os.path.join(video_output_dir, 'background.jpg')

        # Salva o background calculado em um arquivo JPG.
        cv2.imwrite(background_path, background)
        print(f"Background salvo em {background_path}")

    # Libera o vídeo e fecha todas as janelas do OpenCV.
    video.release()
    cv2.destroyAllWindows()

    # Imprime uma mensagem indicando que o processamento do vídeo foi concluído.
    print(f"Processamento do vídeo {video_path} concluído.")


# Diretório contendo os vídeos no Google Drive
videos_dir = os.getcwd() + '\\01-dataset' # mesmo resultado que os.path.join(os.getcwd(), '\01-dataset')

# Diretório de saída principal
output_dir = os.getcwd() + '\\02-output' # mesmo resultado que os.path.join(os.getcwd(), '\02-output')


# Listar todos os arquivos de vídeo na pasta especificada
video_extensions = ('.mov', '.mp4', '.avi', '.mkv')
video_paths = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.lower().endswith(video_extensions)]

# Processa cada vídeo
for video_path in video_paths:
    print(f"Iniciando processamento do vídeo {video_path}")
    process_video(video_path, skip_frames=5, output_dir=output_dir)
    print(f"Processamento do vídeo {video_path} finalizado.")