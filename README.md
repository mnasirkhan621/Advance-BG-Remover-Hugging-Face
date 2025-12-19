# BiRefNet Background Removal API

This project hosts the **BiRefNet** model for high-quality background removal, wrapped in a FastAPI application. It is containerized with Docker for easy deployment on AWS EC2 or App Runner.

## 🧠 Model Details
*   **Architecture**: BiRefNet (Bilateral Reference Network)
*   **Backbone**: `Swin-L` (Swin Transformer Large)
*   **Input Resolution**: 1024x1024 (Standard), Dynamic (Training)
*   **Task**: background removal (DIS5K, COD, HRSOD)

## 🏋️‍♀️ Training Details
The model was configured with the following parameters (extracted from `config.py`):
*   **Dataset**: DIS5K (and others like COD, HRSOD based on config)
*   **Optimizer**: AdamW
*   **Batch Size**: 8 (per GPU)
*   **Precision**: Mixed Precision (`bf16`)
*   **Loss Functions**:
    *   BCE (Binary Cross Entropy)
    *   IoU (Intersection over Union)
    *   SSIM (Structural Similarity)

## 🚀 Deployment (Docker)

### Prerequisites
*   Docker installed on your machine or server.

### 1. Build the Image
```bash
docker build -t birefnet-api .
```

### 2. Run the Container
Run the container and map port 8070:
```bash
docker run -p 8070:8070 birefnet-api
```
*Note: If you are using a GPU instance (like EC2 g4dn), add `--gpus all` to the command.*

### 3. Usage
Once running, open your browser to the interactive Swagger UI:
*   **URL**: [http://localhost:8070/docs](http://localhost:8070/docs)

You can upload an image to the `/remove-bg` endpoint to test it.
