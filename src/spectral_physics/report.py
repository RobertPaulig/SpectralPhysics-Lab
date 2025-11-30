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
