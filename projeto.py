import cv2
import numpy as np
from keras.models import load_model #importando as bibliotecas


video = cv2.VideoCapture(0,cv2.CAP_DSHOW) #Inicializa a camera nativa do pc
model = load_model('Keras_model.h5',compile=False) #Compilando o modelo
data = np.ndarray(shape=(1,224,224,3),dtype=np.float32) #cria um array 4D que armazena as imagens
classes = ["1 brigadeiro","1 brigadeiro"] # Criando as classes para o modelo prever

#Funçao que pre processa a imagem

def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3) #aplica um desfalque gaussiano na imagem
    imgPre = cv2.Canny(imgPre,90,140) #aplica o detecctor de bordas Canny
    kernel = np.ones((4,4),np.uint8) #cria um kernel 4x4
    imgPre = cv2.dilate(imgPre,kernel,iterations=2) #dilata a imagem
    imgPre = cv2.erode(imgPre,kernel,iterations=1) #erosiona a imagem
    return imgPre

#Funçao que detecta o brigadeiro
def DetectarDoce(img):
    imgDoce = cv2.resize(img,(224,224)) #redimensiona a imagem
    imgDoce = np.asarray(imgDoce) #converte a imagem para array
    imgDoceNormalize = (imgDoce.astype(np.float32)/127.0)-1 #normaliza a imagem
    data[0] = imgDoceNormalize #armazena a imagem no array
    prediction = model.predict(data) #faz a previsão usando o modelo carregado do keras
    index = np.argmax(prediction) #encontra o indice da classe
    percent = prediction[0][index] #obtém a porcentagem de confiança da previsão.
    classe = classes[index] #pega o nome da classe correspondente
    return classe,percent

#loop infinito para processar o video

while True:
    _,img = video.read() #lê o frame
    img = cv2.resize(img,(640,480)) #redimensiona a imagem
    imgPre = preProcess(img) #pre processa a imagem
    countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #encontra contornos na imagem

    qtd = 0
    for cnt in countors: #itera sobre os contornos da imagem
        area = cv2.contourArea(cnt) #calcula a area do contorno
        if area > 2000: #filtra contornos com a area maior que 2000 pixels
           x,y,w,h = cv2.boundingRect(cnt) #coordenadas do retângulo delimitador do contorno
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #desenha um retangulo ao redor do contorno da imagem
           recorte = img[y:y +h,x:x+ w] #recorta a regiao do contorno
           classe, conf = DetectarDoce(recorte) #detecta o tipo de doce na regiao recortaada
           cv2.putText(img,str(classe),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2) #escreve o nome da classe na imagem
           if classe == '1 brigadeiro' : qtd+=1 #incrementa +1 na classe

    cv2.rectangle(img,(390,30),(800,80),(0,0,0),-1) #desenha um retangulo preto
    cv2.putText(img, f'Brigadeiro {qtd}', (400, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2) #esreve o texto com a  quantidade de brigadeiros



    cv2.imshow('IMG',img) #mostra a imagem original
    cv2.imshow('IMG PRE', imgPre) #mostra a imagem pre processada
    cv2.waitKey(1) #espera por 1 milissegundo e processa eventos