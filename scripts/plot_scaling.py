import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    data = json.loads(Path("data/reports/bench_scaling.json").read_text(encoding="utf-8"))
    
    chunks = [d["chunks"] / 1000 for d in data]  # In thousands
    mrr = [d["mrr"] for d in data]
    latency = [d["latency_ms"] for d in data]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Tamaño del Corpus (Miles de Chunks)', fontsize=12)
    ax1.set_ylabel('Mean Reciprocal Rank (MRR)', color=color, fontsize=12)
    ax1.plot(chunks, mrr, color=color, marker='o', linewidth=2, markersize=8, label='MRR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Latencia Media de Búsqueda (ms)', color=color, fontsize=12)  
    ax2.plot(chunks, latency, color=color, marker='s', linestyle='--', linewidth=2, markersize=8, label='Latencia')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  
    plt.title("Escalabilidad RAG: Calidad vs Rendimiento (FAISS + BM25)", fontsize=14, pad=15)
    
    # Save the plot
    output_path = Path("docs/scaling_plot.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
