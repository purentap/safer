# Hydra Configuration Guide

This directory contains a modular Hydra configuration structure for managing different model types, input configurations, and training parameters.

## Directory Structure

```
conf/
├── config.yaml              # Main config file
├── model/                   # Model configurations
│   ├── mlp.yaml
│   ├── lstm.yaml            # Generic LSTM (uses inputs.total_dim)
│   ├── lstm_img.yaml        # LSTM for image-only inputs
│   ├── lstm_img_action.yaml # LSTM for img+action inputs
│   └── fusion_lstm.yaml     # Fusion LSTM with adaptive weights
├── inputs/                  # Input configurations
│   ├── img_only.yaml
│   └── img_action.yaml
└── dataset/                 # Dataset configurations
    └── default.yaml
```

## Usage Examples

### 1. LSTM with Image-Only Inputs
```bash
python train.py model=lstm_img inputs=img_only
```

### 2. LSTM with Image + Action Inputs
```bash
python train.py model=lstm_img_action inputs=img_action
```

### 3. Fusion LSTM (with adaptive learnable weights)
```bash
python train.py model=fusion_lstm inputs=img_action
```

### 4. MLP Model
```bash
python train.py model=mlp inputs=img_only
```

### 5. Override Specific Parameters
```bash
# Change learning rate and batch size
python train.py model=lstm_img inputs=img_only training.learning_rate=0.0001 training.batch_size=32

# Disable adaptive weights in FusionLSTM
python train.py model=fusion_lstm inputs=img_action model.params.use_gate=false

# Change LSTM hidden dimension
python train.py model=lstm_img inputs=img_only model.params.hidden_dim=512
```

### 6. Multi-run (Hyperparameter Sweeps)
```bash
# Sweep over learning rates
python train.py -m model=lstm_img inputs=img_only training.learning_rate=0.001,0.0001,0.00001

# Sweep over multiple configurations
python train.py -m model=lstm_img,lstm_img_action inputs=img_only,img_action
```

## Configuration Composition

The configuration uses Hydra's composition feature:

1. **Model Config**: Defines the model architecture (class_name, type, hyperparameters)
2. **Inputs Config**: Defines which inputs to use and their dimensions
3. **Dataset Config**: Defines dataset paths and settings
4. **Main Config**: Combines everything with training hyperparameters

## Key Features

- **Modular Design**: Easy to add new models or input configurations
- **Parameter Resolution**: Input dimensions automatically resolve based on input config
- **Type Safety**: Each model config specifies its class_name for dynamic instantiation
- **Flexible Overrides**: Any parameter can be overridden from command line

## Model Types

1. **MLPModel**: Simple MLP for image embeddings
2. **LSTMModel**: LSTM that can work with image-only or concatenated img+action
3. **FusionLSTMModel**: LSTM with adaptive learnable weights (gate mechanism) for img+action fusion

## Input Configurations

- **img_only**: Uses only image embeddings (2176 dim)
- **img_action**: Uses both image (2176 dim) and action (4096 dim) embeddings

