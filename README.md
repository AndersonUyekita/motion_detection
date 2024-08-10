# Detector de Movimento

Este repositório cria um script em Python para detectar movimento em vídeos gravados. O objetivo é capturar todos os frames onde há um animal.

## 1. Requisitos

Utilizaremos bibliotecas _open source_, desta vez será usada o `OpenCV`.

### 1.1. Preparativos

Instalação do OpenCV (`cv2`).

### 1.2. Bibliotecas

Vamos usar os seguintes packages:

* `cv2`
* `pandas`
* `os`
* `numpy`

## 2. Funcionamento

O _Script_ foi construído baseado em funções.

### 2.1. `get_motion_mask`

* **Objetivo:** A função `get_motion_mask` processa uma máscara de primeiro plano (foreground mask, fg_mask) para destacar regiões onde há movimento detectado, reduzindo ruídos e melhorando a clareza da detecção.

A função `get_motion_mask` é responsável por processar a máscara de movimento bruta (fg_mask), aplicando operações de limiarização, suavização e morfologia para destacar áreas de movimento relevantes e minimizar falsos positivos devido a ruídos. O resultado é uma máscara binária mais precisa que pode ser usada para detecção e rastreamento de objetos em movimento em vídeos.

### 2.2. `non_max_suppression`

* **Objetivo:** Realizar a supressão de não-máximos (NMS, Non-Maximum Suppression) em um conjunto de caixas delimitadoras (bounding boxes) para eliminar detecções redundantes e manter apenas as detecções mais confiáveis.

A função `non_max_suppression` é uma técnica essencial em visão computacional para eliminar detecções múltiplas de um mesmo objeto, mantendo apenas a detecção mais confiável. Isso é particularmente útil em tarefas como detecção de objetos, onde várias caixas delimitadoras podem se sobrepor, representando o mesmo objeto, mas com diferentes níveis de confiança.


### 2.3. `process_video`

* **Objetivo:** Processar um vídeo, detectar movimento em frames específicos, aplicar supressão de não-máximos (NMS) para eliminar detecções redundantes e salvar os frames onde o movimento foi detectado.

A função `process_video` é uma abordagem robusta para processar vídeos, detectando movimento, aplicando técnicas de supressão de não-máximos, e salvando os frames e um background calculado a partir do vídeo. Esta função é ideal para aplicações que exigem monitoramento de movimento em vídeos e processamento de detecção de objetos.