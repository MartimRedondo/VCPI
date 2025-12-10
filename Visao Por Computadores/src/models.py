import torch
import torch.nn as nn
import time
import sys

from constants import device

class ModernCNN(nn.Module):
    """
    Rede Neural Convolucional moderna para classificação de imagens.
    Implementa uma arquitetura com 4 blocos convolucionais seguidos de camadas densas,
    utilizando BatchNorm, Dropout e MaxPooling para regularização e redução da dimensão da imagem.
    
    Parameters:
    - num_classes (int): Número de classes para classificação
    
    Returns:
    Tensor com logits de saída para cada classe (shape: [batch_size, num_classes])
    """
    def __init__(self, num_classes):
        super().__init__()
        # Initial Conv Block
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        # Fourth Conv Block
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    """
    Bloco residual inspirado no ResNet para melhorar o fluxo de gradientes.
    Implementa conexões de salto (skip connections) para permitir treino de redes profundas.
    
    Parameters:
    - in_channels (int): Número de canais de entrada
    - out_channels (int): Número de canais de saída
    - stride (int): Stride da primeira convolução (default: 1)
    
    Returns:
    Tensor com features processadas incluindo conexão residual
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return torch.relu(out)

class ImprovedCNN(nn.Module):
    """
    CNN robusto para classificação GTSRB utilizando técnicas modernas.
    Implementa arquitetura residual com stem layer, blocos residuais hierárquicos
    e Global Average Pooling para reduzir overfitting.
    
    Parameters:
    - num_classes (int): Número de classes para classificação (default: 43 para GTSRB)
    
    Returns:
    Tensor com logits de saída para cada classe (shape: [batch_size, num_classes])
    """
    def __init__(self, num_classes=43):
        super().__init__()
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EfficientCNN(nn.Module):
    """
    CNN eficiente inspirado no EfficientNet para classificação GTSRB.
    Utiliza convoluções separáveis em profundidade (depthwise separable convolutions)
    para reduzir significativamente o número de parâmetros, com a mesma performance.
    
    Parameters:
    - num_classes (int): Número de classes para classificação (default: 43 para GTSRB)
    
    Returns:
    Tensor com logits de saída para cada classe (shape: [batch_size, num_classes])
    """
    def __init__(self, num_classes=43):
        super().__init__()
        
        # Depthwise separable convolutions
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            self._depthwise_block(32, 64, 1),
            self._depthwise_block(64, 128, 2),
            self._depthwise_block(128, 128, 1),
            self._depthwise_block(128, 256, 2),
            self._depthwise_block(256, 256, 1),
            self._depthwise_block(256, 512, 2),
            
            # Final conv
            nn.Conv2d(512, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
    
    def _depthwise_block(self, in_channels, out_channels, stride):

        """
        Cria um bloco de convolução separável em profundidade.
        Combina convolução depthwise (por canal) com pointwise (1x1) para eficiência.
        
        Parameters:
        - in_channels (int): Canais de entrada
        - out_channels (int): Canais de saída
        - stride (int): Stride da convolução depthwise
        
        Returns:
        nn.Sequential contendo o bloco depthwise separable
        """
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, 
                    groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AttentionCNN(nn.Module):
    """
    CNN com mecanismo de atenção espacial para classificação GTSRB.
    Implementa um módulo de atenção que permite ao modelo focar em regiões
    importantes da imagem, melhorando a interpretabilidade e performance.
    
    Parameters:
    - num_classes (int): Número de classes para classificação (default: 43 para GTSRB)
    
    Returns:
    Tensor com logits de saída para cada classe (shape: [batch_size, num_classes])
    """
    def __init__(self, num_classes=43):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        return self.classifier(attended_features)

def train_model(model, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler=None):
    """
    Treina uma rede neural por várias épocas usando conjuntos de treino e validação.

    Parameters:
    - model: modelo a ser treinado (instância de nn.Module)
    - train_loader: DataLoader com os dados de treino
    - val_loader: DataLoader com os dados de validação
    - epochs: número de épocas para treinar
    - loss_fn: função de perda
    - optimizer: otimizador (ex: Adam, SGD)
    - scheduler: agendador opcional de taxa de aprendizagem

    Returns:
    - history: dicionário com perdas e accuracy de treino/validação por época
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\n{'-'*30} Epoch {epoch+1}/{epochs} {'-'*30}")
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, 'best_model.pt')
        
        print(f'Total time: {time.time() - start_time:.2f}s'
              f'\nTrain Loss: {train_loss/len(train_loader):.4f} Train Acc: {train_acc:.2f}%'
              f'\nVal Loss: {val_loss/len(val_loader):.4f} Val Acc: {val_acc:.2f}%')
        # flush print buffer
        sys.stdout.flush()
    
    return history

def evaluate_model(model, test_loader):
    """
    Avalia a accuracy do modelo no conjunto de teste.

    Parameters:
    - model: modelo treinado (instância de nn.Module)
    - test_loader: DataLoader com os dados de teste

    Returns:
    - accuracy: accuracy do modelo (em %)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
