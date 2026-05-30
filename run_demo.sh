#!/bin/bash
echo "Iniciando MITEX Assistant (Modo Demo)"
echo "-----------------------------------"
if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
else
    source .venv/bin/activate
fi

echo "Iniciando chat interactivo (cargando indices de 1M de documentos...)"
mitex chat --index-dir data/indexes/default --topics-report data/reports/topics.json
