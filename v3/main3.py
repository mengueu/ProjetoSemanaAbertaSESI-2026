import cv2
import mediapipe as mp
import numpy as np
import math

# 1. Inicializando o MediaPipe
mp_hands = mp.solutions.hands
# AGORA DETECTA 2 MÃOS
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) 
mp_draw = mp.solutions.drawing_utils

# Captura da webcam
cap = cv2.VideoCapture(0)
success, img = cap.read()
h, w, c = img.shape

# Canvas e variáveis de desenho
canvas = np.zeros((h, w, c), np.uint8)
xp, yp = 0, 0
draw_color = (255, 0, 255) # Cor inicial (Rosa)
brush_size = 15
eraser_size = 50
current_action = "" # Guarda qual ação está sendo feita para não ligar as linhas errado

# 2. Configurando a Paleta de Cores (Menu Superior)
# Cores em formato BGR: Vermelho, Verde, Azul, Amarelo
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] 
color_rects = []
step = w // len(colors)
for i in range(len(colors)):
    # Adiciona as coordenadas de cada retângulo (x_inicial, y_inicial, x_final, y_final)
    color_rects.append((i * step, 0, (i + 1) * step, 80)) 

print("🎥 Câmera iniciada! Pressione 'q' para sair.")

while True:
    success, img = cap.read()
    if not success: break
    
    # Inverte a imagem (efeito espelho)
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # 3. Desenhando o Menu de Cores no topo da tela
    for i, rect in enumerate(color_rects):
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), colors[i], cv2.FILLED)
        
    if results.multi_hand_landmarks:
        # Pega as informações de cada mão e qual lado ela pertence
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Como espelhamos a imagem, o MediaPipe inverte os lados nativamente.
            # Portanto, o que ele chama de 'Left' é a nossa Mão Direita real.
            hand_label = handedness.classification[0].label
            user_hand = "Mão Direita" if hand_label == "Left" else "Mão Esquerda"
            
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                
            if len(lm_list) != 0:
                # 4. Verificando quais dedos estão levantados
                fingers = []
                # Lógica do polegar (Move para os lados, não para cima/baixo)
                if user_hand == "Mão Direita":
                    fingers.append(1 if lm_list[4][1] < lm_list[3][1] else 0)
                else:
                    fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
                    
                # Lógica dos outros 4 dedos (Movem para cima/baixo)
                tip_ids = [8, 12, 16, 20]
                for i in range(4):
                    fingers.append(1 if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2] else 0)

                # ==========================================
                # CONTROLES DA MÃO DIREITA (DESENHAR/APAGAR)
                # ==========================================
                if user_hand == "Mão Direita":
                    
                    # A) MÃO ABERTA: Apaga a tela inteira
                    if fingers[1:] == [1, 1, 1, 1]: 
                        canvas = np.zeros((h, w, c), np.uint8)
                        xp, yp = 0, 0
                        
                    # B) APENAS INDICADOR: Modo Desenho
                    elif fingers[1:] == [1, 0, 0, 0]: 
                        if current_action != "draw": xp, yp = 0, 0 # Reseta se trocou de modo
                        current_action = "draw"
                        
                        x1, y1 = lm_list[8][1], lm_list[8][2]
                        cv2.circle(img, (x1, y1), brush_size, draw_color, cv2.FILLED)
                        if xp == 0 and yp == 0: xp, yp = x1, y1
                        
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_size)
                        xp, yp = x1, y1
                        
                    # C) APENAS POLEGAR: Borracha Local (Desenha na cor preta)
                    elif fingers == [1, 0, 0, 0, 0]: 
                        if current_action != "erase": xp, yp = 0, 0
                        current_action = "erase"
                        
                        x1, y1 = lm_list[4][1], lm_list[4][2]
                        # Feedback visual da borracha na câmera (círculo com borda preta)
                        cv2.circle(img, (x1, y1), eraser_size, (255, 255, 255), 2) 
                        if xp == 0 and yp == 0: xp, yp = x1, y1
                        
                        # O preto "apaga" porque as áreas pretas do canvas ficam transparentes na mesclagem
                        cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_size)
                        xp, yp = x1, y1
                        
                    else:
                        xp, yp = 0, 0
                        
                # ==========================================
                # CONTROLES DA MÃO ESQUERDA (TAMANHO/COR)
                # ==========================================
                elif user_hand == "Mão Esquerda":
                    x_ind, y_ind = lm_list[8][1], lm_list[8][2]
                    x_pol, y_pol = lm_list[4][1], lm_list[4][2]
                    
                    # Se o indicador esquerdo estiver na área do topo da tela
                    if y_ind < 80:
                        # Círculo visual mostrando que você está interagindo com o menu
                        cv2.circle(img, (x_ind, y_ind), 15, (255, 255, 255), cv2.FILLED)
                        for i, rect in enumerate(color_rects):
                            if rect[0] < x_ind < rect[2]:
                                draw_color = colors[i]
                    else:
                        # Se não estiver no menu, calcula o tamanho do bico (Pinaçnado com Polegar e Indicador)
                        cv2.line(img, (x_pol, y_pol), (x_ind, y_ind), (255, 255, 255), 3)
                        
                        # Calcula a distância entre a ponta do polegar e do indicador
                        length = math.hypot(x_ind - x_pol, y_ind - y_pol)
                        
                        # Converte a distância (20 a 200) para o tamanho do lápis (5 a 50)
                        brush_size = int(np.interp(length, [20, 200], [5, 50]))
                        
                        # Mostra uma bolinha flutuante no meio dos dedos representando o tamanho atual do lápis
                        cx, cy = (x_pol + x_ind) // 2, (y_pol + y_ind) // 2
                        cv2.circle(img, (cx, cy), brush_size, draw_color, cv2.FILLED)

    # 5. Mesclagem do Desenho com a Imagem da Câmera
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Tudo que não for preto no canvas (cores) virá uma máscara para não sobrepor a câmera errada
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Quadro Interativo - Semana Aberta", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()