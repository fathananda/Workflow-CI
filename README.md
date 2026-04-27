# Workflow-CI - Iris Classification

**Nama**: Fathi Ananda Mas'ud  
**Email**: fathiananda00@gmail.com  
**Username Dicoding**: fatho_ananda_masud  

---

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
└── MLProject/
    ├── MLProject           # MLflow Project definition
    ├── conda.yaml          # Environment dependencies
    ├── modelling.py        # Training script (dengan argparse)
    └── iris_preprocessing.csv  # Dataset preprocessed
```

---

## Cara Menjalankan MLProject Lokal

```bash
cd Workflow-CI

# Run dengan default parameters
mlflow run MLProject --env-manager=local

# Run dengan custom parameters
mlflow run MLProject --env-manager=local \
  -P n_estimators=200 \
  -P max_depth=7
```

## CI/CD

Workflow `.github/workflows/ci.yml` akan:
1. Setup Python 3.12
2. Install dependencies
3. Jalankan `mlflow run`
4. Upload artefak ke GitHub Actions
5. (Advanced) Build & push Docker image ke Docker Hub

## Secrets yang Dibutuhkan

Tambahkan di Settings > Secrets & Variables > Actions:
- `DOCKER_USERNAME` - Username Docker Hub
- `DOCKER_PASSWORD` - Password/Token Docker Hub
