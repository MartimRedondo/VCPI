import torch

"""
Ficheiro de constantes para configuração do treinamento de modelos de deep learning.
Define parâmetros globais utilizados em todo o projeto, incluindo configurações
de dispositivo, batch size, workers e épocas de treinamento.
"""

# Configuração do dispositivo - utiliza GPU se disponível, caso contrário CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32 # Tamanho do batch para treinamento - número de amostras processadas simultaneamente
WORKERS = 32 # Número de workers para carregamento de dados - processes paralelos para DataLoader
PREFETCH = 8 # Número de batches pré-carregados na memória para otimização
EPOCHS = 20 # Número de épocas de treinamento - quantas vezes o dataset será percorrido
