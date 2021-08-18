import cv2
import mediapipe as mp
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# vincular a webcam ao python
webcam = cv2.VideoCapture(0) # cria a conexão com a webcam

# inicializando o mediapipe
reconhecimento_maos = mp.solutions.hands
desenho_mp = mp.solutions.drawing_utils
maos = reconhecimento_maos.Hands()

# o que aconteceria se ele não tivesse conseguido conectar com a webcam
if webcam.isOpened():
    # vou ler a minha webcam (webcam.read())
    validacao, frame = webcam.read()
    # segundo problema -> entender o que é o webcam.read() -> temos que fazer ele pegar vários frames
    # loop infinito
    while validacao:
        # pegar o próximo frame da tela
        validacao, frame = webcam.read()
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        # converte BGR (padrão do opencv) em RGB (padrão do mediapipe)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame, '@beatrizcrstna', (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (38, 255, 34))
        
        # desenhar a mão
        lista_maos = maos.process(frameRGB)
        if lista_maos.multi_hand_landmarks:
            for mao in lista_maos.multi_hand_landmarks:
                desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)
                
        
        
        # mostrar o frame da webcam que o python ta vendo
        cv2.imshow("Video da Webcam", frame)
        # mandar o python esperar um pouquinho -> de um jeito inteligente
        tecla = cv2.waitKey(2)
        # mandar ele parar o código se eu clicar no ESC
        if tecla == 27:
            break

# primeiro problema -> ele continua conectado na webcam
webcam.release()
cv2.destroyAllWindows()