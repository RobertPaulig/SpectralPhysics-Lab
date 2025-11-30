import argparse
import sys
from .io import load_timeseries_csv, load_spectrum_npz
from .diagnostics import ChannelConfig, SpectralAnalyzer, HealthMonitor


def main():
    """
    CLI для одноканальной спектральной диагностики (норма/аномалия).
    """
    parser = argparse.ArgumentParser(
        description="Одноканальная спектральная диагностика (норма/аномалия).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--signal",
        required=True,
        help="Путь к CSV с сигналом"
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Путь к .npz файлу эталонного спектра"
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Шаг по времени (сек)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Порог аномалии"
    )
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Индекс колонки сигнала в CSV"
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help="Мин. частота анализа (Гц)"
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="Макс. частота анализа (Гц)"
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="Окно для FFT ('hann' или 'none')"
    )
    
    args = parser.parse_args()

    try:
        # 1. Загрузка сигнала
        print(f"Загрузка сигнала из {args.signal}...")
        signal = load_timeseries_csv(args.signal, column=args.column, skip_header=True)
        print(f"  Загружено {len(signal)} отсчётов")

        # 2. Конфиг канала и анализ
        window_value = None if args.window.lower() == "none" else args.window
        
        config = ChannelConfig(
            name="channel-0",
            dt=args.dt,
            window=window_value,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        )
        
        print(f"Анализ спектра (dt={args.dt}, window={window_value})...")
        analyzer = SpectralAnalyzer(config=config)
        current_spectrum = analyzer.analyze(signal)
        print(f"  Спектр содержит {len(current_spectrum.omega)} частотных компонент")

        # 3. Загрузка эталонного спектра
        print(f"Загрузка эталонного спектра из {args.ref}...")
        reference = load_spectrum_npz(args.ref)
        print(f"  Эталонный спектр содержит {len(reference.omega)} частотных компонент")

        # 4. Мониторинг
        print(f"Сравнение с эталоном (threshold={args.threshold})...")
        monitor = HealthMonitor(reference=reference, threshold=args.threshold)
        score = monitor.score(current_spectrum)
        is_anom = monitor.is_anomalous(current_spectrum)

        # 5. Вывод результата
        print("\n" + "=" * 60)
        status = "ANOMALY" if is_anom else "OK"
        print(f"Status: {status}")
        print(f"Score (L2 distance): {score:.6f}")
        print(f"Threshold: {args.threshold:.6f}")
        print("=" * 60)
        
        # Exit code: 0 для OK, 1 для ANOMALY
        return 1 if is_anom else 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
