# CycleGAN Real2Cartoon - VCPI 2024/2025

**Implementação otimizada de CycleGAN para transformação Real↔Cartoon**  
*Martim Redondo (57889) - Universidade do Minho*

## 📁 Estrutura do Projeto

├── organize_datasets.ipynb        # Balanceamento de datasets
├── cyclegan_baseline.ipynb        # Implementação e treinamento
├── advanced_evaluation.ipynb      # Avaliação quantitativa
├── models_complex/cyclegan_*/     # Checkpoints (.pth files)
├── docs/                          # Documentação
└── README.md

## Reprodução Completa

1. **Preparação dos Datasets**: 
   - Baixar os datasets dos links em `docs/relatorio.pdf`
   - Executar `organize_datasets.ipynb` para balanceamento
2. **Treinamento**: Executar `cyclegan_baseline.ipynb`
3. **Avaliação**: Executar `advanced_evaluation.ipynb`

## ⚠️ **Importante - Funcionalidade Completa Preservada**

**✅ Todas as funcionalidades operam normalmente** mesmo sem as epoches e o dataset igual ao usado por mim (o aluno):
- Modelos treinados (25 épocas) carregam automaticamente dos checkpoints
- Avaliação quantitativa funciona completamente
- Geração de amostras opera com qualidade completa
- Continuação de treinamento disponível da época 25 (contudo, será ligeiramente diferente, pq ele continuará a testar com o dataset que foi criado, que será diferente,  mas **mantém a mesma qualidade e balanceamento**)

## 💾 Sistema de Checkpoint Robusto

- **Recuperação**: Carrega `latest_checkpoint.pth` automaticamente ao reiniciar
- **Cópia Total**: Estados dos modelos, otimizadores, schedulers e histórico
- **Limpeza automática**: Mantém apenas os 3 checkpoints mais recentes
- **Best model tracking**: Salva automaticamente o melhor modelo baseado em loss

## 💾 Componentes Incluídos

### Datasets (`dataset/`)
Datasets originais reorganizados e balanceados para treino eficiente.
*Fontes: CelebA-HQ (Kaggle) + Google Cartoon Set (Kaggle)*

### Modelos (`models_complex/`)
- Checkpoints completos (25 épocas)
- Sistema de recuperação automática  
- Histórico de training curves

----

### Notes : o dataset não deu para colocar, nem o best_model.pth e o latest_checkpoint_pth

### Avaliação (`evaluation_results/`)
Métricas quantitativas pré-calculadas (dispensável - regenerável via notebook).

---
**Técnicas Implementadas**: LSGAN, Spectral Normalization, TTUR, Reflection Padding
