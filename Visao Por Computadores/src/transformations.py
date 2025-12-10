import torch
from torchvision.transforms import v2

"""
Transformações de dados para augmentation e pré-processamento de imagens.
Define diferentes configurações de data augmentation para treinamento,
desde transformações básicas até configurações agressivas, visando melhorar
a generalização dos modelos e reduzir overfitting.
"""


# Transformação padrão para treinamento com augmentation moderado
train_transform = v2.Compose([
    v2.Resize((32, 32)),                                                    # Redimensiona para tamanho consistente
    v2.RandomHorizontalFlip(p=0.5),                                         # Flip horizontal aleatório
    v2.RandomRotation(degrees=15),                                          # Rotação aleatória até 15 graus
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variação de cor
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),                       # Translação aleatória
    v2.ToImage(),                                                           # Converte para formato de imagem
    v2.ToDtype(torch.float32, scale=True)                                   # Normaliza para [0,1]
])

# Configuração 1: Augmentation Básica
# Transformações mínimas para casos sensíveis a mudanças geométricas
train_transform_basic = v2.Compose([
    v2.Resize((32, 32)),                    # Redimensiona para tamanho consistente
    v2.RandomHorizontalFlip(p=0.5),         # Apenas flip horizontal
    v2.ToImage(),                           # Converte para formato de imagem        
    v2.ToDtype(torch.float32, scale=True)   # Normaliza para [0,1]
])

# Configuração 2: Augmentation Geométrica
# Foco em transformações espaciais e geométricas
train_transform_geometric = v2.Compose([
    v2.Resize((32, 32)),                                                    # Redimensiona para tamanho consistente 
    v2.RandomHorizontalFlip(p=0.5),                                         # Flip horizontal aleatório                               
    v2.RandomRotation(degrees=30),                                          # Rotação aleatória até 30 graus   
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),    # Transformação afim
    v2.RandomPerspective(distortion_scale=0.2, p=0.5),                      # Distorção de perspectiva
    v2.ToImage(),                                                           # Converte para formato de imagem                     
    v2.ToDtype(torch.float32, scale=True)                                   # Normaliza para [0,1]
])

# Configuração 3: Augmentation de Cor/Intensidade
# Foco em variações de cor, brilho e intensidade
train_transform_color = v2.Compose([
    v2.Resize((32, 32)),                                                    # Redimensiona para tamanho consistente     
    v2.RandomHorizontalFlip(p=0.5),                                         # Flip horizontal aleatório
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Variação intensa de cor
    v2.RandomGrayscale(p=0.1),                                              # Conversão ocasional para escala de cinza        
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),                    # Ajuste de nitidez      
    v2.ToImage(),                                                           # Converte para formato de imagem
    v2.ToDtype(torch.float32, scale=True)                                   # Normaliza para [0,1]
])

# Configuração 4: Augmentation Agressiva (Todas as técnicas)
# Combinação de todas as transformações para máxima variabilidade
train_transform_aggressive = v2.Compose([
    v2.Resize((32, 32)),                                                    # Redimensiona para tamanho consistente                 
    v2.RandomHorizontalFlip(p=0.7),                                         # Flip horizontal aleatório, com maior probabilidade
    v2.RandomVerticalFlip(p=0.3),                                           # Flip vertical adicional            
    v2.RandomRotation(degrees=45),                                          # Rotação aleatória até 45 graus   
    v2.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)),  # Transformação afim intensa
    v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),  # Variação máxima de cor
    v2.RandomPerspective(distortion_scale=0.3, p=0.5),                      # Distorção de perspectiva mais agressiva
    v2.RandomGrayscale(p=0.2),                                              # Conversão ocasional para escala de cinza    
    v2.RandomAdjustSharpness(sharpness_factor=3, p=0.5),                    # Ajuste de nitidez intenso        
    v2.ToImage(),                                                           # Converte para formato de imagem                             
    v2.ToDtype(torch.float32, scale=True)                                   # Normaliza para [0,1]  
])

# Configuração 5: Augmentation com Filtros Avançados
# Inclui técnicas modernas como Random Erasing
train_transform_advanced = v2.Compose([
    v2.Resize((32, 32)),                                                    # Redimensiona para tamanho consistente
    v2.RandomHorizontalFlip(p=0.5),                                         # Flip horizontal aleatório                  
    v2.RandomRotation(degrees=15),                                          # Rotação aleatória até 15 graus                 
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Variação de cor moderada
    v2.ToImage(),                                                           # Converte para formato de imagem                 
    v2.ToDtype(torch.float32, scale=True),                                  # Normaliza para [0,1]     
    v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))           # Aplica Random Erasing com probabilidade de 50%
])

# Transformação para validação/teste - apenas redimensionamento e normalização
test_transform = v2.Compose([
    v2.Resize((32, 32)),                    # Redimensiona para tamanho consistente 
    v2.ToImage(),                           # Converte para formato de imagem 
    v2.ToDtype(torch.float32, scale=True)   # Normaliza para [0,1]
])
