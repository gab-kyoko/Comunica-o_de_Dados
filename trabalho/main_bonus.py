import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, lfilter

#Configurações Gerais
ARQUIVOS_DE_AUDIO = ['audio_01.wav', 'audio_02.wav', 'audio_03.wav'] #Caminho dos audios
TAXA_DE_AMOSTRAGEM = 44100  #Qualidade padrão

#CORREÇÃO: Limita a duração do áudio para evitar erros de memória ---
DURACAO_MAXIMA_SEGUNDOS = 3 #Processa os primeiros 3 segundos de cada áudio

#Cada canal será modulado por uma portadora com frequência distinta:
FREQUENCIAS_PORTADORA = [5000, 12000, 18000]

#Parâmetros do filtro passa-baixa para recuperar os sinais (demodulação):
CORTE_FILTRO = 4000
ORDEM_FILTRO = 6

#Funções Auxiliares
def carregar_audio(caminho_arquivo, taxa_amostragem):
    #converte o audio para mono
    try:
        sinal, _ = librosa.load(caminho_arquivo, sr=taxa_amostragem, mono=True)
        return sinal
    except Exception as e:
        #print(f"Erro ao carregar {caminho_arquivo}: {e}")
        return None

def plotar_espectro(sinal, taxa_amostragem, titulo, nome_arquivo=None):
    n = len(sinal)
    yf = np.fft.fft(sinal)
    xf = np.fft.fftfreq(n, 1 / taxa_amostragem)
    
    plt.figure(figsize=(12, 6))
    plt.plot(xf[:n//2], 2.0/n * np.abs(yf[:n//2]))
    plt.title(titulo)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    if nome_arquivo:
        plt.savefig(nome_arquivo)
    plt.show()

def plotar_espectrograma_comparativo(sinal_original, sinal_recuperado, taxa_amostragem, titulo, nome_arquivo=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    #Espectrograma do sinal original:
    ax1.specgram(sinal_original, NFFT=256, Fs=taxa_amostragem, noverlap=128)
    ax1.set_title(f'Original - {titulo}')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Frequência (Hz)')
    
    #Espectrograma do sinal recuperado:
    ax2.specgram(sinal_recuperado, NFFT=256, Fs=taxa_amostragem, noverlap=128)
    ax2.set_title(f'Recuperado - {titulo}')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Frequência (Hz)')
    
    plt.tight_layout()
    if nome_arquivo:
        plt.savefig(nome_arquivo)
    plt.show()

def filtro_passa_baixa(sinal, corte, fs, ordem=5):
    nyquist = 0.5 * fs
    corte_normalizado = corte / nyquist
    b, a = butter(ordem, corte_normalizado, btype='low', analog=False)
    sinal_filtrado = lfilter(b, a, sinal)
    return sinal_filtrado

#Principal 
def main():
    #print("Carregando arquivos de áudio...")
    sinais_originais = [carregar_audio(f, TAXA_DE_AMOSTRAGEM) for f in ARQUIVOS_DE_AUDIO]
    
    if any(s is None for s in sinais_originais):
        print("Erro ao carregar um ou mais arquivos de áudio. Abortando.")
        return

    #Corta os sinais para a duração máxima definida para evitar erro de memória.
    max_amostras = int(DURACAO_MAXIMA_SEGUNDOS * TAXA_DE_AMOSTRAGEM)
    sinais_originais = [s[:max_amostras] for s in sinais_originais]
    
    #Igualamos os comprimentos dos sinais para facilitar a operação conjunta:
    menor_comprimento = min(len(s) for s in sinais_originais)
    sinais_originais = [s[:menor_comprimento] for s in sinais_originais]
    
    array_tempo = np.linspace(0, menor_comprimento / TAXA_DE_AMOSTRAGEM, num=menor_comprimento)

    #print("Modulando os sinais com suas respectivas portadoras...")
    sinais_modulados = []
    for i, sinal in enumerate(sinais_originais):
        portadora = np.cos(2 * np.pi * FREQUENCIAS_PORTADORA[i] * array_tempo)
        sinal_modulado = sinal * portadora
        sinais_modulados.append(sinal_modulado)

    #letra A
    print("\nEspectros dos sinais modulados separadamente:")
    for i, sinal_mod in enumerate(sinais_modulados):
        plotar_espectro(sinal_mod, TAXA_DE_AMOSTRAGEM, f'Espectro do Canal {i+1} Modulado (Portadora: {FREQUENCIAS_PORTADORA[i]} Hz)', f'espectro_modulado_{i+1}.png')

    #print("Multiplexando os sinais (soma dos modulados)...")
    sinal_multiplexado = np.sum(sinais_modulados, axis=0)

    #letra B
    print("\nb) Espectro do sinal multiplexado:")
    plotar_espectro(sinal_multiplexado, TAXA_DE_AMOSTRAGEM, 'Espectro do Sinal Multiplexado', 'espectro_multiplexado.png')

    #Salvando o sinal multiplexado como arquivo WAV: -> letra C
    arquivo_multiplexado = 'audio_multiplexado.wav'
    print(f"\nc) Salvando áudio multiplexado em '{arquivo_multiplexado}'...")
    sf.write(arquivo_multiplexado, sinal_multiplexado, TAXA_DE_AMOSTRAGEM)

    #letra E e D
    print("\nDemultiplexando e filtrando os sinais para recuperar os originais...")
    sinais_recuperados = []
    for i in range(len(sinais_originais)):

        portadora = np.cos(2 * np.pi * FREQUENCIAS_PORTADORA[i] * array_tempo)
        sinal_demodulado = sinal_multiplexado * portadora

        #Filtra o sinal para recuperar o conteúdo de baixa frequência:
        sinal_recuperado = filtro_passa_baixa(sinal_demodulado, CORTE_FILTRO, TAXA_DE_AMOSTRAGEM, ordem=ORDEM_FILTRO)
        sinais_recuperados.append(sinal_recuperado)
        
        arquivo_recuperado = f'audio_recuperado_{i+1}.wav'
        print(f"Salvando áudio recuperado do canal {i+1} em '{arquivo_recuperado}'...")

        #Normalização para evitar distorções na reprodução:
        if np.max(np.abs(sinal_recuperado)) > 0: #Evita divisão por zero se o sinal for silêncio -> tava dando erro sem
            sinal_recuperado_normalizado = sinal_recuperado / np.max(np.abs(sinal_recuperado))
        else:
            sinal_recuperado_normalizado = sinal_recuperado
        sf.write(arquivo_recuperado, sinal_recuperado_normalizado, TAXA_DE_AMOSTRAGEM)

    #letra F
    print("\nf) Comparando os sinais originais com os recuperados por meio dos espectrogramas:")
    for i in range(len(sinais_originais)):
        plotar_espectrograma_comparativo(sinais_originais[i], sinais_recuperados[i], TAXA_DE_AMOSTRAGEM, f'Canal {i+1}', f'comparacao_canal_{i+1}.png')

    #print("\nProcesso concluído com sucesso!") #terminou e deu certo

if __name__ == '__main__':
    main()
