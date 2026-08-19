"""
TELA INICIAL:
    Clique no botao [D] Destro   OU pressione D  -> mão DIREITA desenha, mão ESQUERDA configura
    Clique no botao [C] Canhoto  OU pressione C  -> mão ESQUERDA desenha, mão DIREITA configura
    ESC ou Q                                     -> sair sem abrir o programa

  MÃO QUE DESENHA:
      - Dedo indicador levantado sozinho  -> Desenhar
      - Mao aberta (4 dedos levantados)   -> Borracha local
      - Joia / thumbs-up (so o polegar)   -> Salvar o desenho na galeria

  MÃO QUE CONFIGURA:
      - Indicador no topo da tela          -> Seleciona cor na paleta
      - Pinca (polegar + indicador)        -> Ajusta tamanho do bico do lapis
      - Mao aberta (4 dedos levantados)    -> Apaga TUDO (limpa o canvas inteiro)

  Teclado:
      - Tecla 'c'       -> Limpar a tela toda
      - Tecla 'q' / ESC -> Sair
"""

import time
import math
import os
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

# =================================================================================
# CONFIGURACOES GERAIS
# =================================================================================

LARGURA_CAM = 1280
ALTURA_CAM = 720

# resolução usada SO para a deteccao da mão (menor = mais rapido).
LARGURA_PROCESSAMENTO = 640

COR_PINCEL_INICIAL = (255, 0, 255)
ESPESSURA_PINCEL_INICIAL = 15
RAIO_BORRACHA = 60

MIN_CONFIANCA_DETECCAO = 0.7
MIN_CONFIANCA_RASTREIO = 0.6
COMPLEXIDADE_MODELO = 1

ALPHA_SUAVIZACAO = 0.5            # 0 = suaviza muito (mas atrasa a linha), 1 = sem suavizar (tremido)
SALTO_MAXIMO_PX = 180             # ignora movimentos maiores que isso num unico frame (provavel erro)
FRAMES_PARA_CONFIRMAR_GESTO = 3   # frames seguidos com o mesmo gesto ate ele "valer"
FRAMES_TOLERANCIA_PERDA_MAO = 6   # quantos frames aguenta sem ver a mão antes de resetar o traco

PASTA_GALERIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "galeria")
COOLDOWN_SALVAR = 2.0
COOLDOWN_CLEAR = 1.5

PALETA_CORES = [
    (0, 0, 255),
    (0, 140, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 200, 0),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 255),
]
ALTURA_PALETA = 80

TITULO_JANELA = "Projeto Semana Aberta"


# =================================================================================
# CLASSES DE ESTABILIDADE
# =================================================================================

class FiltroSuavizacao:
    """
    Suaviza um ponto (x, y) ao longo do tempo usando media movel exponencial
    (EMA) e ignora saltos bruscos de posicao num unico frame (provavel erro
    de deteccao, tipo quando o MediaPipe erra a posicao por 1 frame).
    """

    def __init__(self, alpha=ALPHA_SUAVIZACAO, salto_max=SALTO_MAXIMO_PX):
        self.alpha = alpha
        self.salto_max = salto_max
        self.valor = None  # (x, y) em float, ou None se ainda nao ha historico

    def atualizar(self, novo_ponto):
        if self.valor is None:
            self.valor = (float(novo_ponto[0]), float(novo_ponto[1]))
            return self.valor

        distancia = math.hypot(novo_ponto[0] - self.valor[0], novo_ponto[1] - self.valor[1])

        if distancia > self.salto_max:
            # Salto grande demais para 1 frame -> provavel ruido/erro de
            # deteccao. Mantem o valor anterior em vez de "teleportar".
            return self.valor

        nx = self.alpha * novo_ponto[0] + (1 - self.alpha) * self.valor[0]
        ny = self.alpha * novo_ponto[1] + (1 - self.alpha) * self.valor[1]
        self.valor = (nx, ny)
        return self.valor

    def resetar(self):
        self.valor = None


class EstabilizadorDeGesto:
    """
    Evita que o modo (desenhar/apagar/salvar/ocioso) fique "piscando" por
    causa de uma leitura errada de 1 unico frame. So confirma a troca de
    modo depois que o MESMO gesto aparecer 'frames_para_confirmar' vezes
    seguidas.
    """

    def __init__(self, frames_para_confirmar=FRAMES_PARA_CONFIRMAR_GESTO):
        self.frames_para_confirmar = frames_para_confirmar
        self.modo_confirmado = "ocioso"
        self.modo_candidato = "ocioso"
        self.contagem = 0

    def atualizar(self, modo_detectado_neste_frame):
        if modo_detectado_neste_frame == self.modo_candidato:
            self.contagem += 1
        else:
            self.modo_candidato = modo_detectado_neste_frame
            self.contagem = 1

        if self.contagem >= self.frames_para_confirmar:
            self.modo_confirmado = self.modo_candidato

        return self.modo_confirmado

    def resetar(self):
        self.modo_confirmado = "ocioso"
        self.modo_candidato = "ocioso"
        self.contagem = 0


# =================================================================================
# FUNCOES DE APOIO
# =================================================================================

def dedos_levantados(hand_landmarks, mao_label):
    """[polegar, indicador, medio, anelar, minimo], 1 = levantado."""
    lm = hand_landmarks.landmark
    dedos = []

    if mao_label == "Right":
        dedos.append(1 if lm[4].x < lm[3].x else 0)
    else:
        dedos.append(1 if lm[4].x > lm[3].x else 0)

    pontas = [8, 12, 16, 20]
    juntas = [6, 10, 14, 18]
    for ponta, junta in zip(pontas, juntas):
        dedos.append(1 if lm[ponta].y < lm[junta].y else 0)

    return dedos


def mao_do_usuario(hand_label):
    """
    'Left' do MediaPipe = mão DIREITA real (por causa do espelhamento).
    Isso e sempre assim, independente de a pessoa ser destra ou canhota -
    e so a fisica do espelho, nao tem relacao com a escolha da tela inicial.
    """
    return "esquerda" if hand_label == "Left" else "direita"


def identificar_modo_mao_desenho(dedos):
    """Le a lista de dedos da mão QUE DESENHA e devolve o gesto 'cru'."""
    _, indicador, medio, anelar, minimo = dedos

    if indicador == 1 and medio == 0 and anelar == 0 and minimo == 0:
        return "desenhar"
    if indicador == 1 and medio == 1 and anelar == 1 and minimo == 1:
        return "apagar"
    if dedos == [1, 0, 0, 0, 0]:
        return "salvar"
    return "ocioso"


def identificar_modo_mao_config(dedos, y_indicador):
    """Le a lista de dedos da mão QUE CONFIGURA e devolve o gesto 'cru'."""
    _, indicador, medio, anelar, minimo = dedos

    if indicador == 1 and medio == 1 and anelar == 1 and minimo == 1:
        return "limpar_tudo"
    if indicador == 1 and y_indicador < ALTURA_PALETA:
        return "selecionar_cor"
    if indicador == 1:
        return "ajustar_tamanho"
    return "ocioso"


def mesclar_canvas_no_frame(frame, canvas):
    canvas_cinza = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mascara_inv = cv2.threshold(canvas_cinza, 20, 255, cv2.THRESH_BINARY_INV)
    mascara_inv = cv2.cvtColor(mascara_inv, cv2.COLOR_GRAY2BGR)
    frame_sem_area_do_desenho = cv2.bitwise_and(frame, mascara_inv)
    return cv2.bitwise_or(frame_sem_area_do_desenho, canvas)


def salvar_desenho(frame_final):
    os.makedirs(PASTA_GALERIA, exist_ok=True)
    agora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    caminho = os.path.join(PASTA_GALERIA, f"desenho_{agora}.png")
    cv2.imwrite(caminho, frame_final)
    return caminho


def desenhar_paleta(frame, cor_selecionada):
    largura = frame.shape[1]
    passo = largura // len(PALETA_CORES)
    for i, cor in enumerate(PALETA_CORES):
        x1 = i * passo
        x2 = (i + 1) * passo if i < len(PALETA_CORES) - 1 else largura
        cv2.rectangle(frame, (x1, 0), (x2, ALTURA_PALETA), cor, cv2.FILLED)
        if cor == cor_selecionada:
            cv2.rectangle(frame, (x1 + 4, 4), (x2 - 4, ALTURA_PALETA - 4), (255, 255, 255), 3)


def desenhar_hud(frame, modo_atual, fps, cor_pincel, espessura, instrucoes):
    altura, largura = frame.shape[:2]
    cv2.rectangle(frame, (0, altura - 50), (largura, altura), (30, 30, 30), -1)
    cv2.putText(frame, f"Modo: {modo_atual}", (10, altura - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (largura - 130, altura - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cx_bola = largura // 2
    cy_bola = altura - 25
    cv2.circle(frame, (cx_bola, cy_bola), max(espessura // 2, 4), cor_pincel, -1)
    cv2.circle(frame, (cx_bola, cy_bola), max(espessura // 2, 4), (255, 255, 255), 1)
    cv2.putText(frame, f"Tamanho: {espessura}", (cx_bola + espessura // 2 + 10, cy_bola + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # >>> NOVO v7: instrucoes recebidas prontas (ja montadas de acordo com
    # destro/canhoto), em vez de um texto fixo.
    cv2.putText(frame, instrucoes, (10, altura - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)


# =================================================================================
# >>> NOVO v7: TELA INICIAL - ESCOLHA DESTRO / CANHOTO
# =================================================================================

def escolher_lateralidade(captura):
    """
    Mostra uma tela inicial (com a webcam ao fundo, so pra nao ficar uma
    tela preta parada) pedindo pra pessoa escolher se ela e destra ou
    canhota - clicando com o MOUSE em cima do botao, ou pelo TECLADO
    (D/C) como alternativa. Essa escolha decide qual mão fisica vai
    desenhar e qual vai configurar (cor/tamanho/limpar tudo) no resto do
    programa.

    Por que mouse/teclado e nao gesto? Porque essa e a decisao mais
    importante de tudo - se ela sair errada, todo o resto do app fica
    "trocado". Um clique ou uma tecla sao 100% confiaveis; um gesto de
    reconhecimento poderia falhar justamente na pior hora (antes mesmo da
    pessoa saber usar o sistema).

    Retorna "destro", "canhoto" ou None (se a pessoa pediu pra sair).
    """
    # >>> NOVO: guarda a posicao/clique do mouse. Precisa ser algo MUTAVEL
    # (dict) porque o callback do OpenCV roda numa closure separada e so
    # assim consegue "avisar" o loop principal, la embaixo, do que aconteceu.
    mouse = {"pos": None, "clique": None}

    def ao_evento_mouse(evento, x, y, flags, param):
        mouse["pos"] = (x, y)
        if evento == cv2.EVENT_LBUTTONDOWN:
            mouse["clique"] = (x, y)

    cv2.namedWindow(TITULO_JANELA)
    cv2.setMouseCallback(TITULO_JANELA, ao_evento_mouse)

    def dentro_do_botao(pos, p1, p2):
        if pos is None:
            return False
        return p1[0] <= pos[0] <= p2[0] and p1[1] <= pos[1] <= p2[1]

    while True:
        ok, frame = captura.read()
        if ok:
            frame = cv2.flip(frame, 1)
        else:
            # Sem frame da webcam (raro, mas evita travar a tela de escolha)
            frame = np.zeros((ALTURA_CAM, LARGURA_CAM, 3), dtype=np.uint8)

        altura, largura = frame.shape[:2]

        # escurece o fundo pra destacar o texto
        escurecido = frame.copy()
        cv2.rectangle(escurecido, (0, 0), (largura, altura), (10, 10, 10), cv2.FILLED)
        cv2.addWeighted(escurecido, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Escolha uma opção", (largura // 2 - 260, altura // 2 - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

        # >>> NOVO: cantos dos botoes guardados em variaveis - sao usados
        # tanto pra desenhar o retangulo quanto pra checar hover/clique.
        destro_p1 = (largura // 2 - 360, altura // 2 - 90)
        destro_p2 = (largura // 2 - 30, altura // 2 + 90)
        canhoto_p1 = (largura // 2 + 30, altura // 2 - 90)
        canhoto_p2 = (largura // 2 + 360, altura // 2 + 90)

        sobre_destro = dentro_do_botao(mouse["pos"], destro_p1, destro_p2)
        sobre_canhoto = dentro_do_botao(mouse["pos"], canhoto_p1, canhoto_p2)

        # Coluna esquerda: Destro (fica mais claro e ganha borda quando o
        # mouse passa por cima, pra dar feedback visual de "clicavel")
        cor_fundo_destro = (95, 95, 95) if sobre_destro else (60, 60, 60)
        cv2.rectangle(frame, destro_p1, destro_p2, cor_fundo_destro, -1)
        if sobre_destro:
            cv2.rectangle(frame, destro_p1, destro_p2, (0, 255, 0), 3)
        cv2.putText(frame, "[ D ]", (largura // 2 - 300, altura // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(frame, "Destro", (largura // 2 - 300, altura // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Mão direita desenha", (largura // 2 - 340, altura // 2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Mão esquerda configura", (largura // 2 - 340, altura // 2 + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Coluna direita: Canhoto (mesma logica de destaque)
        cor_fundo_canhoto = (95, 95, 95) if sobre_canhoto else (60, 60, 60)
        cv2.rectangle(frame, canhoto_p1, canhoto_p2, cor_fundo_canhoto, -1)
        if sobre_canhoto:
            cv2.rectangle(frame, canhoto_p1, canhoto_p2, (0, 255, 255), 3)
        cv2.putText(frame, "[ C ]", (largura // 2 + 90, altura // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.putText(frame, "Canhoto", (largura // 2 + 90, altura // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Mão esquerda desenha", (largura // 2 + 50, altura // 2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Mão direita configura", (largura // 2 + 50, altura // 2 + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.putText(frame, "Clique num botão, ou pressione D / C  |  ESC ou Q para sair",
                    (largura // 2 - 320, altura - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow(TITULO_JANELA, frame)

        # >>> NOVO: resolve o clique do mouse (se aconteceu algum desde o
        # ultimo frame) contra os retangulos calculados NESTE frame.
        if mouse["clique"] is not None:
            clique = mouse["clique"]
            mouse["clique"] = None
            if dentro_do_botao(clique, destro_p1, destro_p2):
                return "destro"
            if dentro_do_botao(clique, canhoto_p1, canhoto_p2):
                return "canhoto"

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord('d'), ord('D')):
            return "destro"
        if tecla in (ord('c'), ord('C')):
            return "canhoto"
        if tecla in (ord('q'), ord('Q'), 27):  # ESC
            return None


# =================================================================================
# PROGRAMA PRINCIPAL
# =================================================================================

def main():
    captura = cv2.VideoCapture(1)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_CAM)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_CAM)

    if not captura.isOpened():
        print("Não foi possivel abrir a webcam. Verifique se ela está conectada")
        print("e se nenhum outro programa está usando ela.")
        return

    # >>> NOVO v7: tela inicial de escolha, antes de carregar o MediaPipe
    # (nao precisa de deteccao de mão pra essa etapa, entao adiamos o
    # carregamento do modelo pra depois - o programa abre mais rapido).
    lateralidade = escolher_lateralidade(captura)
    if lateralidade is None:
        captura.release()
        cv2.destroyAllWindows()
        return

    # >>> NOVO v7: define qual mão FISICA joga qual PAPEL no resto do app.
    # "mao_desenho"/"mao_config" valem "direita" ou "esquerda" e sao usados
    # em vez dos antigos rotulos fixos "direita"/"esquerda" da v6.
    if lateralidade == "destro":
        mao_desenho, mao_config = "direita", "esquerda"
    else:
        mao_desenho, mao_config = "esquerda", "direita"

    nome_desenho = mao_desenho.upper()
    nome_config = mao_config.upper()
    instrucoes = (
        f"{nome_desenho}: Indicador = Desenhar  Palma = Borracha  Joia = Salvar | "
        f"{nome_config}: Paleta  Pinça = Tamanho  Aberta = Limpar"
    )

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=COMPLEXIDADE_MODELO,
        min_detection_confidence=MIN_CONFIANCA_DETECCAO,
        min_tracking_confidence=MIN_CONFIANCA_RASTREIO,
    )

    canvas = None
    ponto_anterior = None          # ultimo ponto desenhado (linha continua)
    modo_atual = "ocioso"
    cor_pincel = COR_PINCEL_INICIAL
    espessura_pincel = ESPESSURA_PINCEL_INICIAL
    tempo_anterior = time.time()

    # filtros de posicao (um por ponto que rastreamos na mão que desenha)
    filtro_indicador_desenho = FiltroSuavizacao()
    filtro_palma_desenho = FiltroSuavizacao()

    # estabilizadores de gesto (um por papel, independentes entre si)
    estabilizador_desenho = EstabilizadorDeGesto()
    estabilizador_config = EstabilizadorDeGesto()

    # tolerancia a perda momentanea da mão que desenha
    frames_sem_mao_desenho = FRAMES_TOLERANCIA_PERDA_MAO + 1  # comeca "sem mão"

    ultimo_clear = 0.0
    ultimo_salvar = 0.0
    flash_salvo_ate = 0.0

    print("=== Projeto Semana Aberta ===")
    if lateralidade == "destro":
        print("Perfil: DESTRO  -> mão DIREITA desenha  | mão ESQUERDA configura")
    else:
        print("Perfil: CANHOTO -> mão ESQUERDA desenha | mão DIREITA configura")
    print("Mão que desenha : indicador = desenhar / palma aberta = borracha / joia = salvar")
    print("Mão que configura: paleta de cor / pinça = tamanho / mão aberta = limpar tudo")
    print("Teclado          : C = limpar | Q ou ESC = sair")
    print()

    while True:
        ok, frame = captura.read()
        if not ok:
            print("Não consegui ler o frame da câmera.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        altura, largura = frame.shape[:2]

        # deteccao roda numa copia menor do frame (mais rapido). As
        # coordenadas dos landmarks sao normalizadas (0 a 1), entao
        # convertemos pra pixel usando SEMPRE altura/largura do frame
        # ORIGINAL (cheio), nao da copia reduzida usada aqui.
        escala = LARGURA_PROCESSAMENTO / largura
        frame_deteccao = cv2.resize(frame, (LARGURA_PROCESSAMENTO, int(altura * escala)))
        rgb = cv2.cvtColor(frame_deteccao, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        mao_desenho_detectada = False
        modo_config_bruto = "ocioso"

        if resultado.multi_hand_landmarks and resultado.multi_handedness:
            for hand_landmarks, handedness in zip(
                resultado.multi_hand_landmarks, resultado.multi_handedness
            ):
                mao_label = handedness.classification[0].label
                lado = mao_do_usuario(mao_label)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                dedos = dedos_levantados(hand_landmarks, mao_label)

                # ==========================================================
                # MAO QUE DESENHA
                # ==========================================================
                if lado == mao_desenho:
                    mao_desenho_detectada = True
                    frames_sem_mao_desenho = 0

                    modo_desenho_bruto = identificar_modo_mao_desenho(dedos)
                    # so aceita a troca de modo depois de confirmada
                    modo_desenho = estabilizador_desenho.atualizar(modo_desenho_bruto)

                    if modo_desenho == "desenhar":
                        ponta_ind = hand_landmarks.landmark[8]
                        x_bruto = int(ponta_ind.x * largura)
                        y_bruto = int(ponta_ind.y * altura)
                        # suaviza a posicao e ignora saltos bruscos
                        x_f, y_f = filtro_indicador_desenho.atualizar((x_bruto, y_bruto))
                        x, y = int(x_f), int(y_f)

                        cv2.circle(frame, (x, y), espessura_pincel // 2, cor_pincel, -1)

                        if ponto_anterior is not None:
                            cv2.line(canvas, ponto_anterior, (x, y), cor_pincel, espessura_pincel)
                        else:
                            cv2.circle(canvas, (x, y), espessura_pincel // 2, cor_pincel, -1)
                        ponto_anterior = (x, y)
                        filtro_palma_desenho.resetar()

                    elif modo_desenho == "apagar":
                        centro_palma = hand_landmarks.landmark[9]
                        cx_bruto = int(centro_palma.x * largura)
                        cy_bruto = int(centro_palma.y * altura)
                        cx_f, cy_f = filtro_palma_desenho.atualizar((cx_bruto, cy_bruto))
                        cx, cy = int(cx_f), int(cy_f)

                        cv2.circle(canvas, (cx, cy), RAIO_BORRACHA, (0, 0, 0), -1)
                        cv2.circle(frame, (cx, cy), RAIO_BORRACHA, (255, 255, 255), 2)
                        ponto_anterior = None
                        filtro_indicador_desenho.resetar()

                    elif modo_desenho == "salvar":
                        ponto_anterior = None
                        filtro_indicador_desenho.resetar()
                        filtro_palma_desenho.resetar()

                        agora = time.time()
                        if agora - ultimo_salvar > COOLDOWN_SALVAR:
                            imagem_para_salvar = mesclar_canvas_no_frame(frame, canvas)
                            caminho = salvar_desenho(imagem_para_salvar)
                            ultimo_salvar = agora
                            flash_salvo_ate = agora + 1.5
                            print(f"Desenho salvo em: {caminho}")

                    else:  # ocioso
                        ponto_anterior = None
                        filtro_indicador_desenho.resetar()
                        filtro_palma_desenho.resetar()

                # ==========================================================
                # MAO QUE CONFIGURA
                # ==========================================================
                elif lado == mao_config:
                    lm = hand_landmarks.landmark
                    x_ind = int(lm[8].x * largura)
                    y_ind = int(lm[8].y * altura)
                    x_pol = int(lm[4].x * largura)
                    y_pol = int(lm[4].y * altura)

                    modo_config_bruto = identificar_modo_mao_config(dedos, y_ind)
                    modo_config = estabilizador_config.atualizar(modo_config_bruto)

                    if modo_config == "limpar_tudo":
                        agora = time.time()
                        if agora - ultimo_clear > COOLDOWN_CLEAR:
                            canvas = np.zeros_like(frame)
                            ponto_anterior = None
                            filtro_indicador_desenho.resetar()
                            ultimo_clear = agora
                            modo_atual = "LIMPOU TUDO!"
                        cv2.putText(frame, "LIMPAR TUDO", (largura // 2 - 120, altura // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    elif modo_config == "selecionar_cor":
                        cv2.circle(frame, (x_ind, y_ind), 15, (255, 255, 255), cv2.FILLED)
                        passo = largura // len(PALETA_CORES)
                        for i, cor in enumerate(PALETA_CORES):
                            x1 = i * passo
                            x2 = (i + 1) * passo if i < len(PALETA_CORES) - 1 else largura
                            if x1 < x_ind < x2:
                                cor_pincel = cor
                                break

                    elif modo_config == "ajustar_tamanho":
                        cv2.line(frame, (x_pol, y_pol), (x_ind, y_ind), (255, 255, 255), 3)
                        distancia = math.hypot(x_ind - x_pol, y_ind - y_pol)
                        espessura_pincel = int(np.interp(distancia, [20, 250], [3, 60]))
                        espessura_pincel = max(3, min(espessura_pincel, 60))

                        cx = (x_pol + x_ind) // 2
                        cy = (y_pol + y_ind) // 2
                        cv2.circle(frame, (cx, cy), espessura_pincel // 2, cor_pincel, -1)
                        cv2.circle(frame, (cx, cy), espessura_pincel // 2, (255, 255, 255), 2)

        # tolerancia a perda momentanea da mão que desenha. So reseta o
        # traco se a mão ficar sumida por mais de FRAMES_TOLERANCIA_PERDA_MAO
        # frames seguidos, evitando quebrar a linha por causa de 1-2 frames
        # perdidos (ex: um pisca de deteccao, movimento rapido demais).
        if mao_desenho_detectada:
            modo_para_exibir = estabilizador_desenho.modo_confirmado
        else:
            frames_sem_mao_desenho += 1
            if frames_sem_mao_desenho > FRAMES_TOLERANCIA_PERDA_MAO:
                ponto_anterior = None
                filtro_indicador_desenho.resetar()
                filtro_palma_desenho.resetar()
                estabilizador_desenho.resetar()
                modo_para_exibir = f"sem mão {mao_desenho}"
            else:
                # ainda dentro da tolerancia: mantem o ultimo modo na tela
                modo_para_exibir = estabilizador_desenho.modo_confirmado

        if modo_atual != "LIMPOU TUDO!":
            modo_atual = modo_para_exibir
        elif time.time() - ultimo_clear > 1.0:
            modo_atual = modo_para_exibir

        frame_final = mesclar_canvas_no_frame(frame, canvas)
        desenhar_paleta(frame_final, cor_pincel)

        tempo_atual = time.time()
        fps = 1 / (tempo_atual - tempo_anterior) if tempo_atual != tempo_anterior else 0
        tempo_anterior = tempo_atual
        desenhar_hud(frame_final, modo_atual, fps, cor_pincel, espessura_pincel, instrucoes)

        if time.time() < flash_salvo_ate:
            overlay = frame_final.copy()
            cv2.rectangle(overlay, (largura // 2 - 220, altura // 2 - 50),
                          (largura // 2 + 220, altura // 2 + 50), (0, 80, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame_final, 0.3, 0, frame_final)
            cv2.putText(frame_final, "SALVO NA GALERIA!", (largura // 2 - 190, altura // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow(TITULO_JANELA, frame_final)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:
            break
        elif tecla == ord('c'):
            canvas = np.zeros_like(frame)
            ponto_anterior = None
            filtro_indicador_desenho.resetar()

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
