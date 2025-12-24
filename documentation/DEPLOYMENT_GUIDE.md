# Deployment Guide

Production deployment guide for the Financial Explainable Summarization system.

---

## Deployment Options

### 1. REST API Server (Recommended)
- FastAPI + Uvicorn + Docker
- Multi-client support
- Easy to scale

### 2. Python Package
- Direct integration
- `pip install fin-explainable`

### 3. Cloud Function (Serverless)
- AWS Lambda / GCP Functions
- Auto-scaling, pay-per-use

---

## Quick Deploy

### Docker Deployment
```bash
docker build -t fin-summarizer .
docker run -d --gpus all -p 8000:8000 fin-summarizer
```

### Cloud Deployment
- **AWS:** EC2 p3.2xlarge (V100) - $3/hour
- **GCP:** GKE with nvidia-tesla-v100
- **Azure:** NC6s_v3 with V100

---

## Performance

- **Throughput:** 20 docs/min (single) | 120 docs/min (batch)
- **Latency:** 2.9s per document
- **Cost:** $0.42 per 1000 documents (AWS g4dn.xlarge)

---

## Optimization

1. Quantization: 4x smaller, 2x faster
2. Batch processing: 6x throughput
3. Caching: Reduce repeated compute
4. TorchScript: 20% speedup

---

See full guide for FastAPI implementation, monitoring, and security.
