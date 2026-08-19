"""
Controles com DUAS maos:

  MÃO ESQUERDA (desenho):
      - Dedo indicador levantado sozinho  -> Desenhar
      - Mao aberta (4 dedos levantados)   -> Borracha local (segue o centro da palma)

  MAO DIREITA (configuracao):
      - Indicador no topo da tela          -> Seleciona cor na paleta
      - Pinça (polegar + indicador)        -> Ajusta tamanho do bico do lapis
      - Mao aberta (4 dedos levantados)    -> Apaga tudo

  Teclado:
      - Tecla 'c'       -> Limpar a tela toda
      - Tecla 'q' / ESC -> Sair
"""

import time
import math

import cv2
import numpy as np
import mediapipe as mp

# =================================================================================
# CONFIGURACOES GERAIS
# =================================================================================

LARGURA_CAM = 1280
ALTURA_CAM = 720

COR_PINCEL_INICIAL = (255, 0, 255)   # magenta (BGR)
ESPESSURA_PINCEL_INICIAL = 15        # espessura padrão do traço
RAIO_BORRACHA = 60                   # raio do circulo-borracha (mao direita aberta)

MIN_CONFIANCA_DETECCAO = 0.7
MIN_CONFIANCA_RASTREIO = 0.6

# Paleta de cores (BGR) exibida no topo da tela
PALETA_CORES = [
    (0, 0, 255),       # vermelho
    (0, 140, 255),     # laranja
    (0, 255, 255),     # amarelo
    (0, 255, 0),       # verde
    (255, 200, 0),     # ciano
    (255, 0, 0),       # azul
    (255, 0, 255),     # magenta / rosa
    (255, 255, 255),   # branco
]

ALTURA_PALETA = 80   # altura (em px) da faixa de cores no topo


# =================================================================================
# FUNÇÕES DE APOIO
# =================================================================================

def dedos_levantados(hand_landmarks, mao_label):
    """
    Recebe os 21 landmarks de UMA mao e a label ("Right"/"Left") e
    devolve [polegar, indicador, medio, anelar, minimo]  (1 = levantado).

    Indices dos landmarks do MediaPipe (mao):
        4  = ponta do polegar        3  = junta anterior do polegar
        8  = ponta do indicador      6  = junta do indicador
        12 = ponta do medio          10 = junta do medio
        16 = ponta do anelar         14 = junta do anelar
        20 = ponta do minimo         18 = junta do minimo
    """
    lm = hand_landmarks.landmark
    dedos = []

    # Polegar: comparacao pelo eixo X (muda conforme o lado)
    if mao_label == "Right":
        dedos.append(1 if lm[4].x < lm[3].x else 0)
    else:
        dedos.append(1 if lm[4].x > lm[3].x else 0)

    # Indicador, medio, anelar e minimo: ponta acima da junta => levantado
    pontas = [8, 12, 16, 20]
    juntas = [6, 10, 14, 18]
    for ponta, junta in zip(pontas, juntas):
        dedos.append(1 if lm[ponta].y < lm[junta].y else 0)

    return dedos


def mao_do_usuario(hand_label):
    """
    Converte a label do MediaPipe para a mao REAL do usuario.
    Como a imagem e espelhada (flip), o MediaPipe inverte os lados:
      - O que o MP chama de 'Left'  e a mao DIREITA do usuario.
      - O que o MP chama de 'Right' e a mao ESQUERDA do usuario.
    """
    return "direita" if hand_label == "Left" else "esquerda"


def mesclar_canvas_no_frame(frame, canvas):
    """
    Sobrepoe o desenho (canvas) no video (frame) de forma que as
    areas pretas do canvas fiquem transparentes.
    """
    canvas_cinza = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mascara_inv = cv2.threshold(canvas_cinza, 20, 255, cv2.THRESH_BINARY_INV)
    mascara_inv = cv2.cvtColor(mascara_inv, cv2.COLOR_GRAY2BGR)

    frame_sem_area_do_desenho = cv2.bitwise_and(frame, mascara_inv)
    frame_final = cv2.bitwise_or(frame_sem_area_do_desenho, canvas)
    return frame_final


def desenhar_paleta(frame, cor_selecionada):
    """
    Desenha a barra de cores no topo do frame.
    A cor atualmente selecionada recebe uma borda branca mais grossa.
    """
    largura = frame.shape[1]
    passo = largura // len(PALETA_CORES)

    for i, cor in enumerate(PALETA_CORES):
        x1 = i * passo
        x2 = (i + 1) * passo if i < len(PALETA_CORES) - 1 else largura
        cv2.rectangle(frame, (x1, 0), (x2, ALTURA_PALETA), cor, cv2.FILLED)

        # Destaque na cor selecionada
        if cor == cor_selecionada:
            cv2.rectangle(frame, (x1 + 4, 4), (x2 - 4, ALTURA_PALETA - 4),
                          (255, 255, 255), 3)


def desenhar_hud(frame, modo_atual, fps, cor_pincel, espessura):
    """Desenha a barra de informacoes no rodape."""
    altura, largura = frame.shape[:2]

    # Barra inferior
    cv2.rectangle(frame, (0, altura - 50), (largura, altura), (30, 30, 30), -1)

    # Texto de modo e FPS
    cv2.putText(frame, f"Modo: {modo_atual}", (10, altura - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (largura - 130, altura - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Indicador visual da cor + espessura atual
    cx_bola = largura // 2
    cy_bola = altura - 25
    cv2.circle(frame, (cx_bola, cy_bola), max(espessura // 2, 4), cor_pincel, -1)
    cv2.circle(frame, (cx_bola, cy_bola), max(espessura // 2, 4), (255, 255, 255), 1)
    cv2.putText(frame, f"Tamanho: {espessura}", (cx_bola + espessura // 2 + 10, cy_bola + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Instrucoes
    instrucoes = "Dir: Indicador=Desenhar  Palma=Borracha | Esq: Paleta  Pinca=Tamanho  Aberta=Limpar"
    cv2.putText(frame, instrucoes, (10, altura - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


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
        max_num_hands=2,
        min_detection_confidence=MIN_CONFIANCA_DETECCAO,
        min_tracking_confidence=MIN_CONFIANCA_RASTREIO,
    )

    canvas = None
    ponto_anterior = None        # ultimo ponto desenhado (para linhas continuas)
    modo_atual = "ocioso"
    cor_pincel = COR_PINCEL_INICIAL
    espessura_pincel = ESPESSURA_PINCEL_INICIAL
    tempo_anterior = time.time()

    # Cooldown para o gesto de "limpar tudo" (mao esquerda aberta)
    # Evita que limpe repetidamente enquanto a mao fica aberta
    ultimo_clear = 0.0
    COOLDOWN_CLEAR = 1.5  # segundos

    print("=== Desenho por Gestos v4 - Semana Aberta ===")
    print("Tudo pronto! Use as DUAS maos para controlar.")
    print("  Mao DIREITA : indicador = desenhar / palma aberta = borracha")
    print("  Mao ESQUERDA: paleta de cor / pinca = tamanho / mao aberta = limpar tudo")
    print("  Teclado     : C = limpar | Q ou ESC = sair")
    print()

    while True:
        ok, frame = captura.read()
        if not ok:
            print("Nao consegui ler o frame da camera.")
            break

        frame = cv2.flip(frame, 1)   # espelha (efeito espelho)

        if canvas is None:
            canvas = np.zeros_like(frame)

        altura, largura = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        # Flags de modo para este frame
        mao_direita_detectada = False
        modo_direita = "ocioso"

        if resultado.multi_hand_landmarks and resultado.multi_handedness:
            for hand_landmarks, handedness in zip(
                resultado.multi_hand_landmarks, resultado.multi_handedness
            ):
                mao_label = handedness.classification[0].label
                lado = mao_do_usuario(mao_label)

                # Desenha os landmarks na imagem
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                dedos = dedos_levantados(hand_landmarks, mao_label)

                # ==============================================================
                # MAO DIREITA — desenhar / borracha local
                # ==============================================================
                if lado == "direita":
                    mao_direita_detectada = True
                    _, indicador, medio, anelar, minimo = dedos

                    # A) So o indicador levantado -> DESENHAR
                    if indicador == 1 and medio == 0 and anelar == 0 and minimo == 0:
                        modo_direita = "desenhar"

                        ponta_ind = hand_landmarks.landmark[8]
                        x = int(ponta_ind.x * largura)
                        y = int(ponta_ind.y * altura)

                        # Feedback visual: bolinha na ponta do dedo
                        cv2.circle(frame, (x, y), espessura_pincel // 2, cor_pincel, -1)

                        if ponto_anterior is not None:
                            cv2.line(canvas, ponto_anterior, (x, y),
                                     cor_pincel, espessura_pincel)
                        else:
                            cv2.circle(canvas, (x, y),
                                       espessura_pincel // 2, cor_pincel, -1)
                        ponto_anterior = (x, y)

                    # B) Mao aberta (4 dedos) -> BORRACHA LOCAL
                    elif indicador == 1 and medio == 1 and anelar == 1 and minimo == 1:
                        modo_direita = "apagar"

                        centro_palma = hand_landmarks.landmark[9]
                        cx = int(centro_palma.x * largura)
                        cy = int(centro_palma.y * altura)

                        cv2.circle(canvas, (cx, cy), RAIO_BORRACHA, (0, 0, 0), -1)
                        cv2.circle(frame, (cx, cy), RAIO_BORRACHA, (255, 255, 255), 2)
                        ponto_anterior = None

                    # C) Qualquer outra pose -> ocioso
                    else:
                        modo_direita = "ocioso"
                        ponto_anterior = None

                # ==============================================================
                # MAO ESQUERDA — paleta de cor / tamanho / limpar tudo
                # ==============================================================
                elif lado == "esquerda":
                    _, indicador, medio, anelar, minimo = dedos

                    # Coordenadas uteis
                    lm = hand_landmarks.landmark
                    x_ind = int(lm[8].x * largura)
                    y_ind = int(lm[8].y * altura)
                    x_pol = int(lm[4].x * largura)
                    y_pol = int(lm[4].y * altura)

                    # A) MAO ABERTA -> LIMPAR TUDO (com cooldown)
                    if indicador == 1 and medio == 1 and anelar == 1 and minimo == 1:
                        agora = time.time()
                        if agora - ultimo_clear > COOLDOWN_CLEAR:
                            canvas = np.zeros_like(frame)
                            ponto_anterior = None
                            ultimo_clear = agora
                            modo_atual = "LIMPOU TUDO!"
                        # Feedback visual
                        cv2.putText(frame, "LIMPAR TUDO", (largura // 2 - 120, altura // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    # B) INDICADOR no topo da tela -> SELECIONAR COR
                    elif indicador == 1 and y_ind < ALTURA_PALETA:
                        # Feedback visual: circulo branco no dedo
                        cv2.circle(frame, (x_ind, y_ind), 15, (255, 255, 255), cv2.FILLED)

                        passo = largura // len(PALETA_CORES)
                        for i, cor in enumerate(PALETA_CORES):
                            x1 = i * passo
                            x2 = (i + 1) * passo if i < len(PALETA_CORES) - 1 else largura
                            if x1 < x_ind < x2:
                                cor_pincel = cor
                                break

                    # C) PINCA (polegar + indicador) -> AJUSTAR TAMANHO
                    elif indicador == 1:
                        # Linha visual entre polegar e indicador
                        cv2.line(frame, (x_pol, y_pol), (x_ind, y_ind),
                                 (255, 255, 255), 3)

                        distancia = math.hypot(x_ind - x_pol, y_ind - y_pol)

                        # Mapeia distancia (20~250 px) para espessura (3~60 px)
                        espessura_pincel = int(np.interp(distancia, [20, 250], [3, 60]))
                        espessura_pincel = max(3, min(espessura_pincel, 60))

                        # Bolinha no meio dos dois dedos mostrando o tamanho atual
                        cx = (x_pol + x_ind) // 2
                        cy = (y_pol + y_ind) // 2
                        cv2.circle(frame, (cx, cy), espessura_pincel // 2,
                                   cor_pincel, -1)
                        cv2.circle(frame, (cx, cy), espessura_pincel // 2,
                                   (255, 255, 255), 2)

        # Se a mao direita nao foi detectada, reseta o ponto anterior
        if not mao_direita_detectada:
            ponto_anterior = None
            modo_direita = "sem mao direita"

        # Atualiza o modo exibido
        if modo_atual != "LIMPOU TUDO!":
            modo_atual = modo_direita
        else:
            # Mostra "LIMPOU TUDO!" por 1 segundo
            if time.time() - ultimo_clear > 1.0:
                modo_atual = modo_direita

        # Mescla canvas + frame
        frame_final = mesclar_canvas_no_frame(frame, canvas)

        # Desenha paleta e HUD
        desenhar_paleta(frame_final, cor_pincel)
        tempo_atual = time.time()
        fps = 1 / (tempo_atual - tempo_anterior) if tempo_atual != tempo_anterior else 0
        tempo_anterior = tempo_atual
        desenhar_hud(frame_final, modo_atual, fps, cor_pincel, espessura_pincel)

        cv2.imshow("Projeto Semana Aberta v4", frame_final)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:
            break
        elif tecla == ord('c'):
            canvas = np.zeros_like(frame)
            ponto_anterior = None

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
