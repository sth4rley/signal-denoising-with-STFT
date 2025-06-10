#!/usr/bin/env python3
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import medfilt
import os
import gradio as gr

DIRETORIO_SAIDA = "resultados_processamento"
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Parâmetros da STFT
N_FFT = 2048
TAMANHO_SALTO = 512

def plotar_espectrograma(ax, data, sr, titulo):
    img = librosa.display.specshow(librosa.amplitude_to_db(data, ref=np.max),
                                   y_axis='log', x_axis='time', sr=sr,
                                   hop_length=TAMANHO_SALTO, ax=ax)
    ax.set_title(titulo)
    ax.figure.colorbar(img, ax=ax, format='%+2.0f dB')

def plotar_forma_de_onda(ax, sinal, sr, titulo):
    librosa.display.waveshow(sinal, sr=sr, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

def reduzir_ruido(caminho_audio_entrada, duracao_ruido_ms, fator_agressividade, tamanho_filtro_mediana, nome_arquivo_saida):

    # --- ETAPA 1: CARREGAMENTO E ANÁLISE INICIAL ---
    sinal_audio, taxa_amostragem = librosa.load(caminho_audio_entrada, sr=None)
    stft_completa = librosa.stft(sinal_audio, n_fft=N_FFT, hop_length=TAMANHO_SALTO)
    espectrograma_completo, fase = librosa.magphase(stft_completa)


    # --- ETAPA 2: ESTIMATIVA DO PERFIL DE RUÍDO ---

    # ms -> n frames
    duracao_ruido_em_frames = librosa.time_to_frames(duracao_ruido_ms / 1000.0, sr=taxa_amostragem, hop_length=TAMANHO_SALTO)

    # Calcula o perfil de ruído e a potencia media de cada banda de frequencia
    potencia_ruido = np.mean(espectrograma_completo[:, :duracao_ruido_em_frames], axis=1) # calculo do perfil de ruido



    # --- ETAPA 3: CRIAÇÃO E APLICAÇÃO DA MÁSCARA SUAVE ---
    # a implementação anterior (máscara binária) criava artefatos.
    # a máscara suave atenua o ruído de forma proporcional, ficando mais natural

    # ajusta a intensidade da remoção de ruído com um fator de agressividade
    potencia_ruido_ajustada = potencia_ruido * fator_agressividade

    # Máscara de subtração espectral (Wiener-like),
    # Calcula a máscara de atenuação
    # sinal é forte -> máscara tende a 1.
    # ruído domina -> máscara tende a 0.
    # a fórmula `1 - (ruído/sinal)` é uma simplificação do Filtro de Wiener.
    # adicionamos 1e-8 para evitar divisão por zero em trechos de silêncio.
    mascara = 1 - (potencia_ruido_ajustada[:, None] / (espectrograma_completo + 1e-8))
    mascara = np.clip(mascara, 0, 1) # Garante que a máscara esteja entre 0 e 1
    mascara_antes_filtro = mascara.copy() # Cópia para visualização

    if tamanho_filtro_mediana % 2 == 0:
        tamanho_filtro_mediana += 1
    mascara_suavizada = medfilt(mascara, kernel_size=(1, tamanho_filtro_mediana))



    # --- ETAPA 4: RECONSTRUÇÃO DO SINAL LIMPO ---

    # aplica a mascara no espectograma original
    espectrograma_limpo = espectrograma_completo * mascara_suavizada

    # reconstroi o sinal no dominio do tempo usando a transformada inversa
    sinal_limpo = librosa.istft(espectrograma_limpo * fase, hop_length=TAMANHO_SALTO, length=len(sinal_audio))

    # salva o áudio processado
    caminho_audio_saida = os.path.join(DIRETORIO_SAIDA, f"{nome_arquivo_saida}.wav")
    sf.write(caminho_audio_saida, sinal_limpo, taxa_amostragem)



    # --- ETAPA 5: VISUALIZAÇÃO ---
    # Gráfico 1: Formas de Onda (Original vs. Limpo)
    fig_ondas, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plotar_forma_de_onda(ax1, sinal_audio, taxa_amostragem, "1. Forma de Onda Original")
    plotar_forma_de_onda(ax2, sinal_limpo, taxa_amostragem, "2. Forma de Onda Processada")
    fig_ondas.tight_layout()

    # Gráfico 2: Espectrogramas (Original vs. Limpo)
    fig_espectros, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plotar_espectrograma(ax1, espectrograma_completo, taxa_amostragem, "3. Espectrograma Original")
    plotar_espectrograma(ax2, espectrograma_limpo, taxa_amostragem, "4. Espectrograma Processado")
    fig_espectros.tight_layout()

    # Gráfico 3: Máscaras de Ruído (Pura vs. Suavizada)
    fig_mascaras, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    librosa.display.specshow(mascara_antes_filtro, x_axis='time', y_axis='log', sr=taxa_amostragem, hop_length=TAMANHO_SALTO, cmap='viridis', ax=ax1)
    ax1.set_title("5. Máscara de Ruído (Antes da Suavização)")
    im = librosa.display.specshow(mascara_suavizada, x_axis='time', y_axis='log', sr=taxa_amostragem, hop_length=TAMANHO_SALTO, cmap='viridis', ax=ax2)
    ax2.set_title(f"6. Máscara de Ruído Suavizada (Filtro {tamanho_filtro_mediana})")
    fig_mascaras.colorbar(im, ax=[ax1, ax2])
    fig_mascaras.tight_layout()

    return caminho_audio_saida, fig_ondas, fig_espectros, fig_mascaras


iface = gr.Interface(
    fn=reduzir_ruido,
    inputs=[
        gr.Audio(type="filepath", label="Arquivo de Áudio de Entrada (.wav)"),
        gr.Slider(minimum=10, maximum=1000, value=200, step=10, label="Duração para Estimativa de Ruído (ms)"),
        gr.Slider(minimum=1.0, maximum=5.0, value=1.2, step=0.1, label="Fator de Agressividade da Redução"),
        gr.Slider(minimum=1, maximum=21, value=5, step=2, label="Suavização da Máscara (Filtro de Mediana)"),
        gr.Textbox(value="saida_processada", label="Nome Base para Arquivos de Saída")
    ],
    outputs=[
        gr.Audio(label="Áudio Processado"),
        gr.Plot(label="Análise das Formas de Onda"),
        gr.Plot(label="Análise dos Espectrogramas"),
        gr.Plot(label="Visualização das Máscaras de Ruído"),
    ],
    title="Analise de sinais e sistemas 2025.1",
    description="Sistema de redução de ruído em áudio no domínio da frequência, baseado na Transformada de Fourier de Tempo Curto (STFT) ",
    allow_flagging="never",
    css="footer {display: none !important;}"
)

if __name__ == '__main__':
    iface.launch(share=False)
