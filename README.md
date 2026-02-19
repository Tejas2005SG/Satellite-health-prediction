# 🛰️ Satellite Health AI

AI-Powered Satellite Health Monitoring System built with Modal.com GPU training.

## 📋 Overview

This project implements a complete AI pipeline for:
- **Real-time Health Monitoring**: Detect anomalies in satellite telemetry
- **Predictive Maintenance**: Forecast future failures before they happen

## 🏗️ Architecture

```
Modal.com Cloud Infrastructure
├── Volume: satellite-data-vol (10GB) - Datasets
├── Volume: satellite-results-vol (10GB) - Models & Logs
└── GPU Container (A100) - Training & Inference

Local Development
├── modal/ - Deployment scripts
├── src/ - Core modules
├── tests/ - Empirical testing
└── scripts/ - Automation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Modal
modal token new

# Run setup
python scripts/setup.py
```

### 2. Download Datasets

```bash
# Download all datasets (NASA, ESA, NOAA)
python scripts/download_data.py all

# Or download individually
python scripts/download_data.py nasa
python scripts/download_data.py esa
python scripts/download_data.py noaa
```

### 3. Run Tests

```bash
# Setup verification
python tests/test_setup.py

# Data loading tests
python tests/test_data_loading.py

# Model architecture tests
python tests/test_models.py

# Full integration test
python tests/test_integration.py
```

### 4. Train Models

```bash
# Train all models (2-3 hours)
python scripts/run_training.py all

# Or train individually
modal run modal.train_anomaly
modal run modal.train_predictive
```

### 5. Run Inference

```bash
# Anomaly detection
modal run modal.inference::detect_anomalies

# Predictive maintenance
modal run modal.inference::predict_maintenance
```

## 📊 Datasets

- **NASA SMAP/MSL**: Real telemetry with expert-labeled anomalies (~500MB)
- **ESA OPS-SAT**: European Space Agency CubeSat data (~200MB)
- **NOAA Space Weather**: Solar flare and geomagnetic data (~100MB)

Total: ~800MB stored in Modal volumes

## 🤖 Models

### 1. Anomaly Detection (LSTM Autoencoder)
- **Architecture**: LSTM encoder-decoder
- **Input**: 100-timestep sequences
- **Output**: Reconstruction error for anomaly detection
- **Expected F1**: 0.85-0.88

### 2. Predictive Maintenance (LSTM Forecaster)
- **Architecture**: Multi-layer LSTM
- **Input**: 100-timestep sequences
- **Output**: 20-step ahead predictions
- **Expected MAE**: <2.5°C

## 💰 Cost Estimate

With $30 Modal credits:
- **Dataset download**: ~$0.75
- **Anomaly training**: ~$1.50 (45 min)
- **Predictive training**: ~$1.50 (45 min)
- **Total**: ~$4-5
- **Remaining**: ~$25-26 for experiments

## 📁 Project Structure

```
satellite-health-ai/
├── modal/
│   ├── config.py              # App configuration
│   ├── volumes.py             # Volume setup
│   ├── download_datasets.py   # Data download
│   ├── train_anomaly.py       # Anomaly training
│   ├── train_predictive.py    # Predictive training
│   └── inference.py           # Inference functions
├── src/
│   ├── data/
│   │   ├── loaders.py         # Data loading
│   │   └── preprocessors.py   # Data preprocessing
│   ├── models/
│   │   ├── anomaly_detector.py
│   │   └── predictive_model.py
│   └── utils/
│       ├── metrics.py         # Evaluation metrics
│       └── logger.py          # Logging utilities
├── tests/
│   ├── test_setup.py          # Setup verification
│   ├── test_data_loading.py   # Data tests
│   ├── test_models.py         # Model tests
│   └── test_integration.py    # Integration tests
├── scripts/
│   ├── setup.py               # One-time setup
│   ├── download_data.py       # Dataset download
│   └── run_training.py        # Training launcher
└── requirements.txt           # Dependencies
```

## 🧪 Testing

All components include empirical tests:

- **test_setup.py**: Verify Modal connection, volumes, GPU
- **test_data_loading.py**: Validate data integrity and formats
- **test_models.py**: Test model architectures and GPU utilization
- **test_integration.py**: End-to-end pipeline validation

Run all tests:
```bash
python tests/test_setup.py
python tests/test_data_loading.py
python tests/test_models.py
python tests/test_integration.py
```

## 📈 Expected Performance

| Model | Metric | Expected Value |
|-------|--------|----------------|
| Anomaly Detection | F1-Score | 0.85-0.88 |
| Anomaly Detection | False Positive Rate | <5% |
| Predictive Maintenance | MAE (Temperature) | <2.5°C |
| Predictive Maintenance | Prediction Horizon | 15-20 min |

## 🔧 Configuration

Edit `modal/config.py` to customize:
- GPU type (A100, H100, etc.)
- Model hyperparameters
- Volume sizes
- Dataset sources

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python tests/`
4. Submit pull request

## 📞 Support

For issues and questions:
- Check tests: `python tests/test_setup.py`
- Review logs: `modal logs`
- Open an issue on GitHub

## 🎯 Roadmap

- [x] Phase 1: Infrastructure (Current)
- [ ] Phase 2: Data Pipeline
- [ ] Phase 3: Model Training
- [ ] Phase 4: Testing & Optimization
- [ ] Phase 5: Deployment

---

**Built with Modal.com** ☁️🚀
