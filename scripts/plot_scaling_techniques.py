import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    data = json.loads(Path("data/reports/bench_scaling_techniques.json").read_text(encoding="utf-8"))

    chunks = [d["vectors"] / 1000 for d in data]  # In thousands
    series = {
        "flat": ("FAISS Exacto (Flat, float32)", "tab:red", "o"),
        "hnsw": ("FAISS Aproximado (HNSW)", "tab:green", "^"),
        "turbovec": ("TurboVec Exacto (4-bit)", "tab:orange", "s"),
    }

    plt.figure(figsize=(8, 5))

    for engine, (label, color, marker) in series.items():
        if engine not in data[0]["latency_ms"]:
            continue
        latencies = [d["latency_ms"][engine] for d in data]
        plt.plot(
            chunks,
            latencies,
            marker=marker,
            linewidth=2,
            markersize=8,
            label=label,
            color=color,
        )

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
