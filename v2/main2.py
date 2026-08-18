"""
Desenho por Gestos - Semana Aberta

Controles:
    - Dedo indicador levantado sozinho  -> Desenhar
    - Mao aberta (4 dedos levantados)   -> Apagar (a borracha segue o centro da mao)
    - Tecla 'c'                         -> Limpar a tela toda
    - Tecla 'q' ou ESC                  -> Sair

Como funciona (resumo):
    1. A webcam captura o video frame a frame.
    2. O MediaPipe encontra os 21 pontos (landmarks) da mao em cada frame.
    3. Comparamos a posicao da PONTA de cada dedo com a JUNTA logo abaixo dela
       pra saber se o dedo esta esticado (levantado) ou dobrado (abaixado).
    4. Combinando quais dedos estao levantados, decidimos o "modo" atual
       (desenhar, apagar ou ocioso).
    5. O desenho fica guardado em uma camada separada (o "canvas"), que e
       sobreposta ao video em tempo real, como se fosse tinta sobre o vidro.
"""

import time

import cv2
import numpy as np
import mediapipe as mp

# =================================================================================
# CONFIGURACOES GERAIS
# (deixei tudo separado aqui em cima pra facilitar quando forem adicionar
#  novas funcoes, tipo troca de cor, tamanho do pincel variavel, etc.)
# =================================================================================

LARGURA_CAM = 1280
ALTURA_CAM = 720

COR_PINCEL = (255, 0, 255)     # cor do desenho em BGR (magenta) - ponto de extensao futura
ESPESSURA_PINCEL = 8           # espessura do traco
RAIO_BORRACHA = 60             # raio (em pixels) do circulo que apaga

MIN_CONFIANCA_DETECCAO = 0.7   # confianca minima pra considerar que achou uma mao
MIN_CONFIANCA_RASTREIO = 0.6   # confianca minima pra continuar rastreando a mao


# =================================================================================
# FUNCOES DE APOIO
# =================================================================================

def dedos_levantados(hand_landmarks, mao_label):
    """
    Recebe os 21 landmarks (pontos) de UMA mao e a label ("Right"/"Left") e
    devolve uma lista de 5 posicoes: [polegar, indicador, medio, anelar, minimo],
    onde 1 significa "dedo levantado" e 0 significa "dedo abaixado".

    Indices dos landmarks do MediaPipe (mao):
        4  = ponta do polegar        3  = junta anterior do polegar
        8  = ponta do indicador      6  = junta do indicador
        12 = ponta do medio          10 = junta do medio
        16 = ponta do anelar         14 = junta do anelar
        20 = ponta do minimo         18 = junta do minimo
    """
    lm = hand_landmarks.landmark
    dedos = []

    # Polegar se move mais de lado do que pra cima/baixo, entao comparamos o eixo X.
    # A comparacao inverte dependendo de ser mao direita ou esquerda.
    if mao_label == "Right":
        dedos.append(1 if lm[4].x < lm[3].x else 0)
    else:
        dedos.append(1 if lm[4].x > lm[3].x else 0)

    # Indicador, medio, anelar e minimo: se a ponta esta ACIMA da junta (y menor,
    # pois no OpenCV o eixo Y cresce pra baixo), o dedo esta esticado.
    pontas = [8, 12, 16, 20]
    juntas = [6, 10, 14, 18]
    for ponta, junta in zip(pontas, juntas):
        dedos.append(1 if lm[ponta].y < lm[junta].y else 0)

    return dedos


def identificar_modo(dedos):
    """
    Decide o modo atual (desenhar / apagar / ocioso) com base na lista
    [polegar, indicador, medio, anelar, minimo].

    Regra:
        - So o indicador levantado          -> "desenhar"
        - Indicador + medio + anelar + minimo levantados (mao aberta) -> "apagar"
        - Qualquer outra combinacao          -> "ocioso" (nao faz nada)
    """
    _, indicador, medio, anelar, minimo = dedos

    if indicador == 1 and medio == 0 and anelar == 0 and minimo == 0:
        return "desenhar"

    if indicador == 1 and medio == 1 and anelar == 1 and minimo == 1:
        return "apagar"

    return "ocioso"


def mesclar_canvas_no_frame(frame, canvas):
    """
    Sobrepoe o desenho (canvas) no video (frame), como se o desenho fosse
    tinta em cima do vidro da camera, e nao so uma soma de cores.
    """
    canvas_cinza = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mascara_inv = cv2.threshold(canvas_cinza, 20, 255, cv2.THRESH_BINARY_INV)
    mascara_inv = cv2.cvtColor(mascara_inv, cv2.COLOR_GRAY2BGR)

    frame_sem_area_do_desenho = cv2.bitwise_and(frame, mascara_inv)
    frame_final = cv2.bitwise_or(frame_sem_area_do_desenho, canvas)
    return frame_final


def desenhar_hud(frame, modo_atual, fps):
    """Desenha a barra de informacoes no topo e as instrucoes no rodape."""
    altura, largura = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (largura, 40), (30, 30, 30), -1)
    cv2.putText(frame, f"Modo: {modo_atual}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (largura - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    instrucoes = "Indicador = Desenhar | Mao aberta = Apagar | C = Limpar | Q = Sair"
    cv2.putText(frame, instrucoes, (10, altura - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


# =================================================================================
# PROGRAMA PRINCIPAL
# =================================================================================

def main():
    captura = cv2.VideoCapture(0)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_CAM)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_CAM)

    if not captura.isOpened():
        print("Nao foi possivel abrir a webcam. Verifique se ela esta conectada")
        print("e se nenhum outro programa (Zoom, Teams, etc.) esta usando ela.")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_CONFIANCA_DETECCAO,
        min_tracking_confidence=MIN_CONFIANCA_RASTREIO,
    )

    canvas = None
    ponto_anterior = None
    modo_atual = "ocioso"
    tempo_anterior = time.time()

    print("Tudo pronto! Uma janela de video deve abrir.")
    print("Pressione Q ou ESC nela pra sair, e C pra limpar o desenho.")

    while True:
        ok, frame = captura.read()
        if not ok:
            print("Nao consegui ler o frame da camera.")
            break

        frame = cv2.flip(frame, 1)  # espelha o video (fica mais natural, tipo espelho)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks and resultado.multi_handedness:
            hand_landmarks = resultado.multi_hand_landmarks[0]
            mao_label = resultado.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dedos = dedos_levantados(hand_landmarks, mao_label)
            modo_atual = identificar_modo(dedos)

            altura, largura = frame.shape[:2]

            if modo_atual == "desenhar":
                ponta_indicador = hand_landmarks.landmark[8]
                x = int(ponta_indicador.x * largura)
                y = int(ponta_indicador.y * altura)

                if ponto_anterior is not None:
                    cv2.line(canvas, ponto_anterior, (x, y), COR_PINCEL, ESPESSURA_PINCEL)
                else:
                    cv2.circle(canvas, (x, y), ESPESSURA_PINCEL // 2, COR_PINCEL, -1)
                ponto_anterior = (x, y)

            elif modo_atual == "apagar":
                # landmark 9 = base do dedo medio, um bom "centro" estavel da palma
                centro_palma = hand_landmarks.landmark[9]
                cx = int(centro_palma.x * largura)
                cy = int(centro_palma.y * altura)

                cv2.circle(canvas, (cx, cy), RAIO_BORRACHA, (0, 0, 0), -1)
                cv2.circle(frame, (cx, cy), RAIO_BORRACHA, (255, 255, 255), 2)
                ponto_anterior = None

            else:
                ponto_anterior = None
        else:
            ponto_anterior = None
            modo_atual = "sem mao detectada"

        frame_final = mesclar_canvas_no_frame(frame, canvas)

        tempo_atual = time.time()
        fps = 1 / (tempo_atual - tempo_anterior) if tempo_atual != tempo_anterior else 0
        tempo_anterior = tempo_atual

        desenhar_hud(frame_final, modo_atual, fps)

        cv2.imshow("Desenho por Gestos - Semana Aberta", frame_final)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:  # ESC
            break
        elif tecla == ord('c'):
            canvas = np.zeros_like(frame)

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
