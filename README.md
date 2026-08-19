# Projeto Semana Aberta

Projeto de **Visão Computacional em Python** que transforma a webcam num quadro de desenho: você desenha no ar, com as mãos, sem tocar em nada. Feito pelos alunos do curso técnico de Desenvolvimento de Sistemas para ser apresentado na Semana Aberta.

---

## O que é a Semana Aberta?

A **Semana Aberta** é um evento realizado para os alunos de curso técnico SENAI apresentarem seus trabalhos para os demais alunos da escola, de fora do curso, uma chance de mostrar, na prática, o que é produzido em sala de aula ao longo do curso.

Neste projeto, o público-alvo da apresentação são os **alunos do Fundamental 2**, então a proposta foi pensada para ser **lúdica, visual e rápida de entender**, sem depender de nenhuma explicação técnica prévia.

## O que é o nosso projeto

Um programa em Python que usa a câmera do computador para detectar as mãos da pessoa em tempo real e permite **desenhar**: o dedo indicador vira "caneta", a mão aberta vira borracha, e gestos simples trocam a cor e o tamanho do traço, tudo sem mouse, teclado ou caneta física, só com as mãos.

## Propósito do projeto

A ideia surgiu da vontade de apresentar algo de **Visão Computacional** de um jeito acessível para crianças do Fundamental 2. Em vez de uma explicação técnica (que poderia intimidar ou entediar essa faixa etária), escolhemos uma demonstração **interativa e divertida**: a própria criança usa as mãos para desenhar na tela e vê, na hora, o computador "entendendo" seus gestos.

Para acompanhar a demonstração, preparamos também uma **apresentação básica sobre Visão Computacional** — o que é, pra que serve, onde aparece no dia a dia — mantida propositalmente simples, sem entrar em termos técnicos que pudessem afastar o público.

---

## Tecnologias e ferramentas utilizadas

| Ferramenta | Papel no projeto |
|---|---|
| **Python 3** | Linguagem principal do projeto |
| **OpenCV** (`opencv-python`) | Captura de vídeo da webcam, desenho na tela (linhas, círculos, texto, retângulos), exibição da janela e captura de mouse/teclado |
| **MediaPipe** (`mediapipe`, solução *Hands*) | Detecção da mão em tempo real: encontra os 21 pontos (landmarks) de cada mão e identifica se é a mão esquerda ou direita |
| **NumPy** (`numpy`) | Manipulação da imagem como matriz — criação e mesclagem do "canvas" (camada onde o desenho fica guardado) e mapeamento de distância → espessura do traço |
| **math** (biblioteca padrão) | Cálculo de distância entre pontos (usado no gesto de "pinça" que ajusta o tamanho do lápis) |
| **time** (biblioteca padrão) | Cálculo de FPS e controle de *cooldown* de gestos (evitar disparo repetido) |
| **os / datetime** (biblioteca padrão) | Criação automática da pasta `galeria/` e nomeação dos desenhos salvos com data e hora |

---

## Evolução do projeto (v1 → v7)

O grupo evoluiu o projeto em 7 versões, cada uma resolvendo uma limitação ou acrescentando um recurso da anterior. Esse histórico é justamente o que este README documenta:

| Versão | Arquivo | Principais novidades |
|---|---|---|
| **v1** | `main1.py` | Primeira versão funcional (prova de conceito): 1 mão, código simples e direto (sem funções). Indicador levantado desenha; mão aberta (sem contar o polegar) apaga a tela inteira. Cor e espessura do traço fixas. |
| **v2** | `main2.py` | Código reorganizado em funções (mais legível e fácil de estender). "Mão aberta" deixa de apagar tudo e passa a ser uma **borracha local** (um círculo que segue o centro da palma). Ganha uma barra de status (modo atual + FPS) e instruções na tela. Tecla `C` passa a ser o atalho para limpar tudo. |
| **v3** | `main3.py` | Salto de recurso: o programa passa a reconhecer **as duas mãos ao mesmo tempo**. Nasce a divisão de papéis entre as mãos: uma mão desenha, a outra configura. Surge a **paleta de cores** (4 cores) selecionável com a mão de configuração, e o ajuste de espessura por gesto de **pinça** (distância entre polegar e indicador). |
| **v4** | `main4.py` | Reorganização do código nos moldes limpos da v2, mas já com as duas mãos da v3. Os papéis das mãos são refinados e fixados: a mão de desenho passa a ter **borracha local** na mão aberta (como na v2), enquanto **limpar tudo** e a **paleta** (agora com 8 cores) vão para a mão de configuração — essa divisão criada aqui é a que o projeto usa até a v7. Adiciona *cooldown* para o gesto de limpar tudo não disparar várias vezes seguidas. |
| **v5** | `main5.py` | Novo gesto: **joia (👍, só o polegar levantado)** na mão de desenho salva automaticamente o desenho atual (já mesclado com a imagem da câmera) como uma imagem PNG dentro de uma pasta `galeria/`, com nome baseado em data/hora e um aviso visual "SALVO NA GALERIA". |
| **v6** | `main6.py` | Versão de **robustez**: a detecção passa a rodar numa cópia reduzida do frame (processa mais rápido sem perder precisão) e a resolução é reajustada para um ponto ideal entre qualidade e desempenho. Introduz filtros de **suavização de movimento** (média móvel + rejeição de saltos bruscos) e um **estabilizador de gesto** que só troca de modo depois que o mesmo gesto se repete por alguns frames seguidos — reduzindo bastante o "tremido" e as trocas de modo por engano. Também passa a tolerar a mão sumir da câmera por instantes sem quebrar o traço. |
| **v7** | `main7.py` | Adiciona uma **tela inicial de escolha "Destro / Canhoto"**, com botões clicáveis pelo mouse (destacados ao passar o cursor por cima) e atalhos de teclado como alternativa. A partir dessa escolha, o programa **inverte dinamicamente** qual mão desenha e qual configura — tornando a experiência confortável também para pessoas canhotas, algo importante já que, no evento, dezenas de crianças diferentes vão usar o sistema em sequência. Mantém toda a robustez conquistada na v6. |

> 💡 Em resumo: **v1-v2** validaram a ideia com uma mão só; **v3-v4** expandiram para duas mãos com papéis definidos (desenhar vs. configurar); **v5** deu um "prêmio" ao usuário (salvar o desenho); **v6** deixou o reconhecimento confiável o suficiente para uso ao vivo; e **v7** tornou a experiência acessível para canhotos e destros.

---

## ▶️ Como utilizar o projeto

> ⚠️ **Atenção com a versão do Python:** o projeto usa a API "clássica" do MediaPipe (`mediapipe.solutions.hands`), que é simples e funciona **100% offline**. Só que essa API foi **removida** nas versões mais novas da biblioteca (a partir da 0.10.30), e essas versões novas só têm instalador disponível pra **Python 3.13 ou mais novo**. Ou seja: se você estiver com Python 3.13+, o `pip` vai instalar automaticamente uma versão do `mediapipe` que **não tem** `mediapipe.solutions`, e o programa vai quebrar com o erro `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
>
> **Solução:** use Python 3.10, 3.11 ou 3.12 pra criar o ambiente virtual deste projeto. No Windows, se você já tem várias versões do Python instaladas, pode escolher qual usar com o launcher `py`:
> ```powershell
> py -0p                      # lista as versões de Python instaladas no seu PC
> py -3.11 -m venv venv       # cria o venv usando especificamente o Python 3.11
> venv\Scripts\activate
> ```
> Se você não tem nenhuma versão 3.10–3.12 instalada, baixe em [python.org/downloads](https://www.python.org/downloads/) (qualquer 3.10.x, 3.11.x ou 3.12.x serve).

### 2. Instalação das bibliotecas

Com o ambiente virtual **ativado** (usando uma versão de Python compatível, ver acima):

```bash
pip install opencv-python mediapipe==0.10.14 numpy
```

Isso instala exatamente as versões testadas: `opencv-python`, `mediapipe==0.10.14` e `numpy`.

> Reforçando: O `mediapipe==0.10.14` continua exigindo Python 3.9 a 3.12, seja instalado via `-r requirements.txt` ou via `pip install` direto.

### 3. Executando

A versão recomendada pra apresentação é a `main7.py` (a mais completa e a mais estável no reconhecimento):

```bash
python main7.py
```

Uma janela com a imagem da webcam vai abrir. Se o sistema pedir permissão de câmera, aceite.

> **Atenção:** Se você estiver usando uma única webcam no computador, a função: 
> ```python
> captura = cv2.VideoCapture(1)
> ```
> Deve ser alterada para:
> ```python
> captura = cv2.VideoCapture(0)
> ```
> No caso de exitir 2 webcam, você pode escolher entre os parâmetros 0 ou 1 (0 = primeira câmera; 1 = segunda câmera)

### 3. Tela inicial (só na v7)
Ao abrir o programa, escolha se você é **Destro** ou **Canhoto**, clicando no botão correspondente com o mouse ou pressionando `D` ou `C` no teclado. Essa escolha define qual mão desenha e qual mão configura o resto da sessão. `ESC` ou `Q` nessa tela fecham o programa sem abrir a câmera de desenho.

### 4. Controles dentro do programa (versão final — v7)

**Mão que desenha:**
| Gesto | Ação |
|---|---|
| Só o dedo indicador levantado | Desenha |
| Mão aberta (4 dedos) | Borracha local (apaga onde a palma passar) |
| "Joia" — só o polegar levantado | Salva o desenho atual na pasta `galeria/` |

**Mão que configura:**
| Gesto | Ação |
|---|---|
| Indicador na faixa de cores (topo da tela) | Seleciona a cor do pincel |
| Pinça (aproximar polegar e indicador) | Ajusta a espessura do traço |
| Mão aberta (4 dedos) | Limpa **todo** o desenho |

**Teclado (a qualquer momento):**
| Tecla | Ação |
|---|---|
| `C` | Limpa toda a tela |
| `Q` ou `ESC` | Sai do programa |

> Nas versões anteriores (v1 a v6) os papéis de cada mão são fixos — a mão direita sempre desenha e a esquerda sempre configura. Veja o cabeçalho de cada arquivo `.py` para os detalhes específicos daquela versão.

---

## ⚙️ Como funciona o projeto

O funcionamento é o mesmo em todas as versões, ficando mais sofisticado a cada uma. Em linhas gerais:

1. **Captura de vídeo**: o OpenCV lê a webcam quadro a quadro (frame a frame) e espelha a imagem horizontalmente, pra parecer um espelho de verdade.
2. **Detecção da mão**: cada frame é passado para o MediaPipe Hands, que devolve os **21 pontos (landmarks)** de cada mão detectada — pontas dos dedos, juntas, base da palma — além de identificar se é a mão "esquerda" ou "direita" (do ponto de vista da câmera).
3. **Quais dedos estão levantados**: o programa compara a posição da ponta de cada dedo com a posição da junta logo abaixo dela. Se a ponta está mais alta que a junta, o dedo está esticado; senão, está dobrado. O polegar é tratado à parte, pois ele se move mais de lado do que para cima/baixo.
4. **Identificando o gesto e o modo**: combinando quais dedos estão levantados, o programa decide o que fazer — desenhar, apagar, selecionar cor, ajustar tamanho, limpar tudo ou salvar (dependendo da versão).
5. **O "canvas" invisível**: o desenho não é feito direto em cima do vídeo. Ele fica guardado numa camada separada, preta, do tamanho do vídeo (o *canvas*). A cada frame, essa camada é mesclada com a imagem da câmera: as áreas pretas do canvas viram transparentes e só o que foi desenhado aparece por cima do vídeo — como se fosse tinta sobre um vidro.
6. **Interface (HUD)**: por cima de tudo isso, o programa desenha a paleta de cores, o modo atual, a cor/espessura selecionada e o FPS, para dar retorno visual constante de que o sistema está "entendendo" o gesto.
7. **Estabilidade (a partir da v6)**: como o reconhecimento de mão pode falhar ou tremer de um frame para o outro, duas técnicas amenizam isso:
   - **Suavização de posição** (média móvel exponencial): a posição usada para desenhar é uma média entre o ponto atual e os anteriores, e saltos bruscos de um único frame são ignorados (provável erro de detecção).
   - **Confirmação de gesto**: o modo só muda de fato depois que o mesmo gesto aparecer por alguns frames seguidos, evitando que o traço "pisque" entre modos por uma leitura errada isolada.
8. **Seleção de lateralidade (v7)**: antes de tudo isso começar, uma tela pergunta se a pessoa é destra ou canhota. A resposta define, para toda a sessão, qual mão física assume o papel de "mão que desenha" e qual assume "mão que configura".

---

## 👥 Equipe

*Grupo 8: Miguel Augusto Rocha, Ana Luiza Ferreira Antônio, Maria Luiza Aguilar e Pedro Antunes.*
