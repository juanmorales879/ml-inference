# ML Inference API

A production-ready REST API for serving image classification predictions using PyTorch.

## Overview

This API provides endpoints for:
- Image classification with confidence scores
- Model version management
- Prediction logging and monitoring
- Health checks and metrics

## Tech Stack

- **Framework**: FastAPI
- **ML Framework**: PyTorch
- **Model**: ResNet-50 (pre-trained on ImageNet)
- **Containerization**: Docker
- **Deployment**: Kubernetes

## Project Status

🚧 **In Development**

Current version: 0.1.0

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ml-inference-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights
bash scripts/download_model.sh
