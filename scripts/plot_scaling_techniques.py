import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    data = json.loads(Path("data/reports/bench_scaling_techniques.json").read_text(encoding="utf-8"))
    
    chunks = [d["vectors"] / 1000 for d in data]  # In thousands
    flat_lat = [d["latency_ms"]["flat"] for d in data]
    hnsw_lat = [d["latency_ms"]["hnsw"] for d in data]
    turbovec_lat = [d["latency_ms"]["turbovec"] for d in data]
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(chunks, flat_lat, marker='o', linewidth=2, markersize=8, label='FAISS Exacto (Flat, float32)', color='tab:red')
    plt.plot(chunks, turbovec_lat, marker='s', linewidth=2, markersize=8, label='TurboVec Exacto (4-bit)', color='tab:orange')
    plt.plot(chunks, hnsw_lat, marker='^', linewidth=2, markersize=8, label='FAISS Aproximado (HNSW)', color='tab:green')
    
    plt.xlabel('Tamaño del Corpus (Miles de Vectores)', fontsize=12)
    plt.ylabel('Latencia de Búsqueda Semántica (ms)', fontsize=12)
    plt.title("Comparativa de Arquitecturas: Latencia vs Tamaño", fontsize=14, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.tight_layout()  
    
    # Save the plot
    output_path = Path("docs/scaling_techniques_plot.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
