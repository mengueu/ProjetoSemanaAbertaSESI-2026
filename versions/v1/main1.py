import cv2
import mediapipe as mp
import numpy as np

# Inicializando o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Lendo o primeiro frame para pegar as dimensões da tela
success, img = cap.read()
h, w, c = img.shape

# Criando um "quadro em branco" onde os desenhos vão ficar salvos
canvas = np.zeros((h, w, c), np.uint8)

# Variáveis para guardar a posição anterior do dedo (para traçar a linha)
xp, yp = 0, 0

print("Câmera iniciada! Pressione 'q' para sair.")

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Inverter a imagem (efeito espelho) para ficar mais intuitivo
    img = cv2.flip(img, 1)
    
    # O MediaPipe usa o padrão de cor RGB, e o OpenCV usa BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Se uma mão for detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha os pontos e conexões da mão na tela
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Pegando as coordenadas de todos os pontos da mão
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                
            if len(lm_list) != 0:
                # IDs das pontas dos dedos: Indicador(8), Médio(12), Anelar(16), Mindinho(20)
                # IDs das articulações abaixo da ponta: 6, 10, 14, 18
                tip_ids = [8, 12, 16, 20]
                pip_ids = [6, 10, 14, 18]
                
                fingers = []
                # Checa quais dedos estão levantados (comparando a altura da ponta com a articulação)
                for i in range(4):
                    if lm_list[tip_ids[i]][2] < lm_list[pip_ids[i]][2]:
                        fingers.append(1) # Dedo levantado
                    else:
                        fingers.append(0) # Dedo abaixado
                        
                # ----------------------------------------------------
                # MODO DESENHO: Apenas o indicador levantado
                # ----------------------------------------------------
                if fingers == [1, 0, 0, 0]:
                    x1, y1 = lm_list[8][1], lm_list[8][2]
                    
                    # Círculo na ponta do dedo para feedback visual
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    
                    # Se for o primeiro toque, a posição inicial é a atual
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                        
                    # Desenha uma linha no nosso canvas
                    cv2.line(canvas, (xp, yp), (x1, y1), (255, 0, 255), 10)
                    xp, yp = x1, y1
                    
                # ----------------------------------------------------
                # MODO APAGAR: Todos os dedos levantados (Mão Aberta)
                # ----------------------------------------------------
                elif fingers == [1, 1, 1, 1]:
                    # Zera o canvas (preenche de preto novamente)
                    canvas = np.zeros((h, w, c), np.uint8)
                    xp, yp = 0, 0
                    
                else:
                    # Se não estiver nem desenhando nem apagando, reseta as posições
                    xp, yp = 0, 0

    # Mesclando o Canvas (desenho) com o frame da Câmera (imagem real)
    # Transforma o canvas em cinza e cria uma máscara
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    # Aplica a máscara e junta as imagens para cores vivas
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    # Mostra o resultado final na tela
    cv2.imshow("Projeto Semana Aberta v1", img)
    
    # Pressione 'q' no teclado para fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()