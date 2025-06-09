import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import medfilt
import os

DIRETORIO_SAIDA = "resultados_processamento"
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Parâmetros da STFT
# A STFT divide o sinal em pequenas janelas e aplica a FFT em cada uma.
N_FFT = 2048          # número de pontos da FFT.
TAMANHO_SALTO = 512   # de quantas em quantas amostras a janela da FFT se move.


def reduzir_ruido(caminho_audio_entrada, duracao_ruido_ms, tamanho_filtro_mediana, nome_arquivo_saida):

    # --- ETAPA 1: CARREGAMENTO E ANÁLISE INICIAL ---
    sinal_audio, taxa_amostragem = librosa.load(caminho_audio_entrada, sr=None) # (sr=None preserva a taxa de amostragem original)

    stft_completa = librosa.stft(sinal_audio, n_fft=N_FFT, hop_length=TAMANHO_SALTO)

    espectrograma_completo, fase = librosa.magphase(stft_completa) # separa a stft em magnitude e fase


    # --- ETAPA 2: ESTIMATIVA DO PERFIL DE RUÍDO ---

    # converte: ms -> numero de frames
    duracao_ruido_em_frames = librosa.time_to_frames(duracao_ruido_ms / 1000.0, sr=taxa_amostragem, hop_length=TAMANHO_SALTO)

    # Calcula o perfil de ruído e a potencia media de cada banda de frequencia
    potencia_ruido = np.mean(espectrograma_completo[:, :duracao_ruido_em_frames], axis=1)


    # --- ETAPA 3: CRIAÇÃO E APLICAÇÃO DA MÁSCARA BINÁRIA ---
    # para cada ponto (frequencia, tempo) no espectograma, se a magnitude for maior que a
    # potencia média do ruído para aquela frequência, consideramos true
    mascara_ruido = espectrograma_completo > potencia_ruido[:, None] # O [:, None] Expande o array para permitir a comparação com toda matriz
    mascara_ruido = mascara_ruido.astype(float)  # Converte True/False para 1.0/0.0
    mascara_antes_filtro_para_plot = mascara_ruido.copy() # cópia para visualização

    # Suaviza a máscara com um filtro de mediana.
    if tamanho_filtro_mediana % 2 == 0: # O filtro de mediana precisa de um tamanho ímpar.
        tamanho_filtro_mediana += 1
    mascara_ruido = medfilt(mascara_ruido, kernel_size=(1, tamanho_filtro_mediana))


    # --- ETAPA 4: RECONSTRUÇÃO DO SINAL LIMPO ---

    # Aplica a máscara ao espectrograma original, zerando as partes identificadas como ruído.
    espectrograma_limpo = espectrograma_completo * mascara_ruido

    # Reconstrói o sinal no domínio do tempo usando a Transformada Inversa (iSTFT).
    sinal_limpo = librosa.istft(espectrograma_limpo * fase, hop_length=TAMANHO_SALTO) # Espectograma limpo * Fase Original

    # Salva o áudio processado em um arquivo .wav.
    caminho_audio_saida = os.path.join(DIRETORIO_SAIDA, f"{nome_arquivo_saida}.wav")
    sf.write(caminho_audio_saida, sinal_limpo, taxa_amostragem)


    # --- ETAPA 5: VISUALIZAÇÃO ---

    plt.figure(figsize=(15, 20))

    # Espectrograma Original
    plt.subplot(5, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(espectrograma_completo, ref=np.max),
                             y_axis='log', x_axis='time', sr=taxa_amostragem, hop_length=TAMANHO_SALTO)
    plt.colorbar(format='%+2.0f dB')
    plt.title('1. Espectrograma Original (Sinal + Ruído)')

    # Região de Ruído (zoom nos primeiros X ms)
    plt.subplot(5, 1, 2)
    espectrograma_ruido = espectrograma_completo[:, :duracao_ruido_em_frames]
    librosa.display.specshow(librosa.amplitude_to_db(espectrograma_ruido, ref=np.max),
                                 y_axis='log', x_axis='time', sr=taxa_amostragem, hop_length=TAMANHO_SALTO)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'2. Região Usada para Estimar o Ruído (primeiros {duracao_ruido_ms}ms)')

    # Máscara Binária Pura (antes do filtro)
    plt.subplot(5, 1, 3)
    librosa.display.specshow(mascara_antes_filtro_para_plot, x_axis='time', y_axis='log', sr=taxa_amostragem, hop_length=TAMANHO_SALTO, cmap='gray')
    plt.colorbar()
    plt.title('3. Máscara de Ruído Binária (Antes da Suavização)')

    # Máscara Após Suavização
    plt.subplot(5, 1, 4)
    librosa.display.specshow(mascara_ruido, x_axis='time', y_axis='log', sr=taxa_amostragem, hop_length=TAMANHO_SALTO, cmap='gray')
    plt.colorbar()
    plt.title(f'4. Máscara de Ruído Suavizada (Filtro de Mediana com kernel {tamanho_filtro_mediana})')

    # Espectrograma Final Processado
    plt.subplot(5, 1, 5)
    librosa.display.specshow(librosa.amplitude_to_db(espectrograma_limpo, ref=np.max),
                             y_axis='log', x_axis='time', sr=taxa_amostragem, hop_length=TAMANHO_SALTO)
    plt.colorbar(format='%+2.0f dB')
    plt.title('5. Espectrograma Processado (Sinal Limpo)')

    # Ajusta os plots para não se sobreporem
    plt.tight_layout()
    caminho_plot_analise = os.path.join(DIRETORIO_SAIDA, f"{nome_arquivo_saida}_analise_espectral.png")
    plt.savefig(caminho_plot_analise, dpi=300, bbox_inches='tight')
    plt.close()

    return caminho_audio_saida, caminho_plot_analise


audio_saida, analise_saide = reduzir_ruido("entrada.wav", 400, 5, "saida.wav")
