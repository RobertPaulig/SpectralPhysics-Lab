from typing import Dict
import datetime

def generate_markdown_report(
    scores: Dict[str, float],
    thresholds: Dict[str, float],
    out_path: str,
    title: str = "Spectral Health Report",
) -> None:
    """
    Сохранить markdown-файл с таблицей по каналам.
    
    Args:
        scores: Словарь {имя_канала: дистанция}.
        thresholds: Словарь {имя_канала: порог}.
        out_path: Путь для сохранения отчета.
        title: Заголовок отчета.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        f"# {title}",
        f"",
        f"**Date:** {timestamp}",
        f"",
        f"## Channel Status",
        f"",
        f"| Channel | Distance | Threshold | Status |",
        f"|---------|----------|-----------|--------|"
    ]
    
    any_anomaly = False
    
    for name, distance in scores.items():
        threshold = thresholds.get(name, 0.0)
        is_anom = distance > threshold
        status = "🔴 **ANOMALY**" if is_anom else "🟢 OK"
        
        if is_anom:
            any_anomaly = True
            
        lines.append(
            f"| `{name}` | {distance:.6f} | {threshold:.6f} | {status} |"
        )
        
    lines.append("")
    
    if any_anomaly:
        lines.append("> [!WARNING]")
        lines.append("> Anomalies detected! Please check the affected channels.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> All systems nominal.")
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def generate_ndt_report(
    profile_ldos: "np.ndarray",
    current_ldos: "np.ndarray",
    scores: "np.ndarray",
    mask: "np.ndarray",
    out_path: str,
    title: str = "Spectral NDT Report"
) -> None:
    """
    Generate NDT report with defect statistics.
    """
    import numpy as np
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    n_defects = np.sum(mask)
    total_pixels = mask.size
    defect_ratio = n_defects / total_pixels * 100
    
    max_score = np.max(scores)
    mean_score = np.mean(scores)
    
    lines = [
        f"# {title}",
        f"",
        f"**Date:** {timestamp}",
        f"",
        f"## Defect Statistics",
        f"- **Defect Pixels:** {n_defects} / {total_pixels} ({defect_ratio:.2f}%)",
        f"- **Max Defect Score:** {max_score:.4f}",
        f"- **Mean Score:** {mean_score:.4f}",
        f"",
        f"## Defect Locations (Top 10)",
        f"| X | Y | Score |",
        f"|---|---|-------|"
    ]
    
    # Find top 10 defects
    if n_defects > 0:
        flat_indices = np.argsort(scores.flatten())[::-1]
        top_indices = flat_indices[:10]
        
        ny, nx = scores.shape
        for idx in top_indices:
            y, x = np.unravel_index(idx, (ny, nx))
            score = scores[y, x]
            if mask[y, x]:
                lines.append(f"| {x} | {y} | {score:.4f} |")
    else:
        lines.append("| - | - | - |")
        
    lines.append("")
    
    if n_defects > 0:
        lines.append("> [!WARNING]")
        lines.append(f"> **{n_defects} defect pixels detected!** Check the map.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> No defects detected.")
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

