import cv2 #OpenCV
import mediapipe as MP

# Inicializando as bibliotecas --------------------------------------
webcam = cv2.VideoCapture(0) # o índice padrão das WebCam é 0
# Importa a solução do MP responsavel por identicar
solucaoIdentificacao = MP.solutions.face_detection
# Elemento para identificar o rosto 
Identificador = solucaoIdentificacao.FaceDetection()
delimitador = MP.solutions.drawing_utils

while True:
    # Obtem a imagem da WebCam
    boolConfirm, imagem = webcam.read() #Ordem padrão

    # Confirma o processamento do frame
    if not boolConfirm: 
        break

    # Obtem em uma lista as imagens de rosto reconhecidas
    faces = Identificador.process(imagem)

    #imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    if faces.detections:
        for face in faces.detections:
            delimitador.draw_detection(imagem, face)

    # Mostra a imagem na tela
    cv2.imshow("Face Detection", imagem)
    # Tempo de espera na tela por segundo e interrompe do programa na tela quando seleconando ESC;
    if cv2.waitKey(2) == 27:
        break

#Encerrando a webcam e a janela no windowns
webcam.release()
cv2.destroyAllWindows()