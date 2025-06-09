# signal-denoising-with-STFT
Sistema de redução de ruído em áudio baseado no domínio da frequência. Utiliza a Transformada de Fourier de Tempo Curto (STFT) para analisar o sinal e estimar o perfil de ruído a partir de um trecho silencioso. Uma máscara binária, suavizada com filtro de mediana, suprime o ruído antes da reconstrução do áudio com a transformada inversa.
