#!/bin/bash
echo "=== Configurando entorno MITEX para compañeros ==="

# 1. Instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download es_core_news_sm
ollama pull gemma3:4b

# 2. Descargar índices (Sustituir por el enlace real o usar gdown)
echo "Descargando índices masivos (3.7 GB)..."
# Ejemplo si usas un link directo de OneDrive/Servidor:
# wget -O MITEX_INDEXES.tar.gz "TU_LINK_DE_DESCARGA_AQUI"
# (Si usas Google Drive, es recomendable instalar 'gdown': pip install gdown && gdown "TU_LINK_DE_DRIVE")

# 3. Extraer en la carpeta correcta
echo "Extrayendo índices..."
tar -xzvf MITEX_INDEXES.tar.gz
rm MITEX_INDEXES.tar.gz

echo "¡Todo listo! Ejecuta: source .venv/bin/activate && mitex chat"
