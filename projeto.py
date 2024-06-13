import cv2
import numpy as np
from keras.models import load_model

video = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Inicializa a câmera USB
model = load_model('Keras_model.h5', compile=False) # Carrega o modelo sem compilar
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # Cria um array 4D que armazena as imagens
classes = ["1 brigadeiro", "1 brigadeiro"] # Classes para o modelo prever

# Função que pré-processa a imagem
def preProcess(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3) # Aplica um desfoque gaussiano na imagem
    imgPre = cv2.Canny(imgPre, 90, 140) # Aplica o detector de bordas Canny
    kernel = np.ones((4, 4), np.uint8) # Cria um kernel 4x4
    imgPre = cv2.dilate(imgPre, kernel, iterations=2) # Dilata a imagem
    imgPre = cv2.erode(imgPre, kernel, iterations=1) # Erosiona a imagem
    return imgPre

# Função que detecta o brigadeiro
def DetectarDoce(img):
    imgDoce = cv2.resize(img, (224, 224)) # Redimensiona a imagem
    imgDoce = np.asarray(imgDoce) # Converte a imagem para array
    imgDoceNormalize = (imgDoce.astype(np.float32) / 127.0) - 1 # Normaliza a imagem
    data[0] = imgDoceNormalize # Armazena a imagem no array
    prediction = model.predict(data) # Faz a previsão usando o modelo carregado do keras
    index = np.argmax(prediction) # Encontra o índice da classe
    percent = prediction[0][index] # Obtém a porcentagem de confiança da previsão
    classe = classes[index] # Pega o nome da classe correspondente
    return classe, percent

# Loop infinito para processar o vídeo
while True:
    _, img = video.read() # Lê o frame
    img = cv2.resize(img, (640, 480)) # Redimensiona a imagem
    imgPre = preProcess(img) # Pré-processa a imagem
    contours, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Encontra contornos na imagem

    qtd = 0
    for cnt in contours: # Itera sobre os contornos da imagem
        area = cv2.contourArea(cnt) # Calcula a área do contorno
        if area > 2000: # Filtra contornos com a área maior que 2000 pixels
            x, y, w, h = cv2.boundingRect(cnt) # Coordenadas do retângulo delimitador do contorno
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # Desenha um retângulo ao redor do contorno da imagem
            recorte = img[y:y+h, x:x+w] # Recorta a região do contorno
            classe, conf = DetectarDoce(recorte) # Detecta o tipo de doce na região recortada
            cv2.putText(img, str(classe), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Escreve o nome da classe na imagem
            if classe == '1 brigadeiro': 
                qtd += 1 # Incrementa +1 na classe

    cv2.rectangle(img, (390, 30), (800, 80), (0, 0, 0), -1) # Desenha um retângulo preto
    cv2.putText(img, f'Brigadeiro {qtd}', (400, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2) # Escreve o texto com a quantidade de brigadeiros

    cv2.imshow('IMG', img) # Mostra a imagem original
    cv2.imshow('IMG PRE', imgPre) # Mostra a imagem pré-processada
    cv2.waitKey(1) # Espera por 1 milissegundo e processa eventos
