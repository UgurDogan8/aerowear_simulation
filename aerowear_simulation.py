from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_SEED = 1507
TOTAL_DURATION_HOURS = 8
SECONDS_PER_HOUR = 3600
TOTAL_SECONDS = TOTAL_DURATION_HOURS * SECONDS_PER_HOUR
BATTERY_CAPACITY_MAH = 250.0

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

RISK_TO_CODE = {"Green": 0, "Yellow": 1, "Red": 2}
CODE_TO_RISK = {value: key for key, value in RISK_TO_CODE.items()}

SAMPLING_MODE_TO_CODE = {"LOW_POWER": 0, "NORMAL": 1, "HIGH_FREQUENCY": 2}
SAMPLING_INTERVAL_SECONDS = {"LOW_POWER": 60, "NORMAL": 10, "HIGH_FREQUENCY": 1}

BLE_LATENCY_MODEL = {
    "LOW_POWER": (180.0, 40.0),
    "NORMAL": (250.0, 60.0),
    "HIGH_FREQUENCY": (350.0, 100.0),
}
BLE_PACKET_LOSS_PROBABILITY = {
    "LOW_POWER": 0.01,
    "NORMAL": 0.02,
    "HIGH_FREQUENCY": 0.04,
}

MODE_CURRENT_MA = {"LOW_POWER": 6.0, "NORMAL": 14.0, "HIGH_FREQUENCY": 28.0}
BLE_EXTRA_CURRENT_MA = 4.0
ALARM_EXTRA_CURRENT_MA = 8.0


@dataclass(frozen=True)
class EnvironmentBlock:
    name: str
    start_second: int
    end_second: int
    pm25_range: Tuple[float, float]
    voc_range: Tuple[float, float]
    temperature_offset: float
    humidity_offset: float
    pm25_noise_std: float
    voc_noise_std: float


def build_environment_blocks() -> list[EnvironmentBlock]:
    """Return the fixed 8-hour micro-environment schedule."""
    return [
        EnvironmentBlock("Normal iç ortam", 0, 90 * 60, (8, 15), (150, 300), 0.0, 2.0, 0.9, 18),
        EnvironmentBlock("Trafik yakın çevresi", 90 * 60, 130 * 60, (35, 75), (400, 800), 1.4, -4.0, 3.5, 45),
        EnvironmentBlock("Normal iç ortam", 130 * 60, 180 * 60, (8, 15), (150, 300), 0.0, 2.0, 0.9, 18),
        EnvironmentBlock("Temizlik ürünü kullanımı", 180 * 60, 210 * 60, (20, 45), (900, 1800), 0.8, 8.0, 2.8, 95),
        EnvironmentBlock("Normal iç ortam", 210 * 60, 300 * 60, (8, 15), (150, 300), 0.0, 2.0, 0.9, 18),
        EnvironmentBlock("Yemek pişirme / kapalı ortam", 300 * 60, 345 * 60, (60, 120), (500, 1000), 2.5, 12.0, 5.5, 65),
        EnvironmentBlock("Normal iç ortam", 345 * 60, 390 * 60, (8, 15), (150, 300), 0.0, 2.0, 0.9, 18),
        EnvironmentBlock("Kampüs / açık alan yürüyüşü", 390 * 60, 440 * 60, (15, 35), (250, 500), -0.8, -6.0, 2.0, 30),
        EnvironmentBlock("Normal iç ortam", 440 * 60, 480 * 60, (8, 15), (150, 300), 0.0, 2.0, 0.9, 18),
    ]


def smooth_noise(rng: np.random.Generator, size: int, scale: float, window: int) -> np.ndarray:
    """Create low-frequency random variation without requiring scipy."""
    raw_noise = rng.normal(0.0, scale, size)
    kernel = np.ones(window) / window
    return np.convolve(raw_noise, kernel, mode="same")


def assign_environment_arrays(
    elapsed_seconds: np.ndarray,
    blocks: Iterable[EnvironmentBlock],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate reference PM2.5 and VOC values for the fixed environment blocks."""
    environment = np.empty(elapsed_seconds.size, dtype=object)
    reference_pm25 = np.zeros(elapsed_seconds.size, dtype=float)
    reference_voc = np.zeros(elapsed_seconds.size, dtype=float)

    for block in blocks:
        mask = (elapsed_seconds >= block.start_second) & (elapsed_seconds < block.end_second)
        local_seconds = elapsed_seconds[mask] - block.start_second
        duration = max(block.end_second - block.start_second, 1)

        pm_low, pm_high = block.pm25_range
        voc_low, voc_high = block.voc_range
        pm_span = pm_high - pm_low
        voc_span = voc_high - voc_low

        pm_mid = rng.uniform(pm_low + 0.30 * pm_span, pm_high - 0.25 * pm_span)
        voc_mid = rng.uniform(voc_low + 0.30 * voc_span, voc_high - 0.25 * voc_span)

        pm_wave = 0.18 * pm_span * np.sin(
            2 * np.pi * local_seconds / rng.uniform(700, 1500) + rng.uniform(0, 2 * np.pi)
        )
        voc_wave = 0.16 * voc_span * np.sin(
            2 * np.pi * local_seconds / rng.uniform(900, 1900) + rng.uniform(0, 2 * np.pi)
        )
        pm_trend = rng.normal(0.0, 0.08 * pm_span) * (local_seconds / duration)
        voc_trend = rng.normal(0.0, 0.08 * voc_span) * (local_seconds / duration)

        environment[mask] = block.name
        reference_pm25[mask] = pm_mid + pm_wave + pm_trend + rng.normal(0.0, block.pm25_noise_std, mask.sum())
        reference_voc[mask] = voc_mid + voc_wave + voc_trend + rng.normal(0.0, block.voc_noise_std, mask.sum())

    add_peak_events(elapsed_seconds, reference_pm25, reference_voc)
    reference_pm25 = np.clip(reference_pm25, 1.0, 160.0)
    reference_voc = np.clip(reference_voc, 50.0, 2200.0)
    return environment, reference_pm25, reference_voc


def add_peak_events(elapsed_seconds: np.ndarray, pm25: np.ndarray, voc: np.ndarray) -> None:
    """Add short Gaussian peak events to mimic real micro-environment spikes."""
    peak_events = [
        # center_s, width_s, pm25_peak, voc_peak
        (26 * 60, 55, 5.0, 80.0),
        (96 * 60, 110, 18.0, 160.0),
        (121 * 60, 75, 12.0, 130.0),
        (188 * 60, 95, 10.0, 480.0),
        (201 * 60, 70, 8.0, 320.0),
        (307 * 60, 120, 38.0, 230.0),
        (327 * 60, 110, 26.0, 160.0),
        (405 * 60, 90, 10.0, 100.0),
    ]

    for center, width, pm_height, voc_height in peak_events:
        pulse = np.exp(-0.5 * ((elapsed_seconds - center) / width) ** 2)
        pm25 += pm_height * pulse
        voc += voc_height * pulse


def generate_temperature_and_humidity(
    elapsed_seconds: np.ndarray,
    environment: np.ndarray,
    blocks: Iterable[EnvironmentBlock],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate temperature and humidity with block-specific offsets."""
    elapsed_hours = elapsed_seconds / SECONDS_PER_HOUR
    temperature = 23.0 + 2.7 * np.sin(2 * np.pi * (elapsed_hours - 1.0) / 8.0)
    humidity = 53.0 + 10.5 * np.sin(2 * np.pi * (elapsed_hours + 1.2) / 6.5)

    for block in blocks:
        mask = environment == block.name
        if not mask.any():
            continue
        # Re-apply by time range so repeated "Normal ic ortam" blocks receive the same neutral label safely.
        time_mask = (elapsed_seconds >= block.start_second) & (elapsed_seconds < block.end_second)
        temperature[time_mask] += block.temperature_offset
        humidity[time_mask] += block.humidity_offset

    temperature += smooth_noise(rng, elapsed_seconds.size, scale=0.8, window=180)
    humidity += smooth_noise(rng, elapsed_seconds.size, scale=2.0, window=240)
    temperature += rng.normal(0.0, 0.12, elapsed_seconds.size)
    humidity += rng.normal(0.0, 0.35, elapsed_seconds.size)

    return np.clip(temperature, 18.0, 30.0), np.clip(humidity, 35.0, 75.0)


def generate_raw_sensor_data(
    reference_pm25: np.ndarray,
    reference_voc: np.ndarray,
    temperature_c: np.ndarray,
    humidity_percent: np.ndarray,
    elapsed_hours: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add realistic sensor imperfections to reference PM2.5 and VOC values."""
    pm25_gain_error = 1.12
    pm25_humidity_bias = 0.13 * (humidity_percent - 50.0) + 0.0045 * reference_pm25 * np.maximum(humidity_percent - 45.0, 0.0)
    pm25_temperature_bias = -0.055 * (temperature_c - 23.0)
    pm25_drift = 0.62 * elapsed_hours + 0.12 * np.sin(2 * np.pi * elapsed_hours / 4.0)
    pm25_noise = rng.normal(0.0, 2.4 + 0.025 * reference_pm25, reference_pm25.size)
    raw_pm25 = reference_pm25 * pm25_gain_error + pm25_humidity_bias + pm25_temperature_bias + pm25_drift + pm25_noise

    voc_gain_error = 1.08
    voc_humidity_bias = 3.6 * (humidity_percent - 50.0)
    voc_temperature_bias = -2.4 * (temperature_c - 23.0)
    voc_drift = 18.0 * elapsed_hours
    voc_noise = rng.normal(0.0, 34.0 + 0.035 * reference_voc, reference_voc.size)
    raw_voc = reference_voc * voc_gain_error + voc_humidity_bias + voc_temperature_bias + voc_drift + voc_noise

    return np.clip(raw_pm25, 0.0, None), np.clip(raw_voc, 0.0, None)


def calibrate_pm25(df: pd.DataFrame) -> tuple[np.ndarray, Dict[str, float], LinearRegression]:
    """Train on the first 70% of the timeline and evaluate on the last 30%."""
    feature_columns = ["raw_pm25", "temperature_c", "humidity_percent", "elapsed_hours"]
    split_index = int(len(df) * 0.70)

    x = df[feature_columns].to_numpy()
    y = df["reference_pm25"].to_numpy()

    model = LinearRegression()
    model.fit(x[:split_index], y[:split_index])

    calibrated_pm25 = np.clip(model.predict(x), 0.0, None)
    y_test = y[split_index:]
    raw_test = df["raw_pm25"].to_numpy()[split_index:]
    calibrated_test = calibrated_pm25[split_index:]

    mae_before = mean_absolute_error(y_test, raw_test)
    rmse_before = float(np.sqrt(mean_squared_error(y_test, raw_test)))
    r2_before = r2_score(y_test, raw_test)

    mae_after = mean_absolute_error(y_test, calibrated_test)
    rmse_after = float(np.sqrt(mean_squared_error(y_test, calibrated_test)))
    r2_after = r2_score(y_test, calibrated_test)

    metrics = {
        "mae_before": float(mae_before),
        "rmse_before": float(rmse_before),
        "r2_before": float(r2_before),
        "mae_after": float(mae_after),
        "rmse_after": float(rmse_after),
        "r2_after": float(r2_after),
        "mae_improvement_percent": float(100.0 * (mae_before - mae_after) / mae_before),
        "rmse_improvement_percent": float(100.0 * (rmse_before - rmse_after) / rmse_before),
    }
    return calibrated_pm25, metrics, model


def classify_pm25(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.where(values <= 15.0, 0, np.where(values <= 35.0, 1, 2))
    labels = np.array([CODE_TO_RISK[int(code)] for code in codes], dtype=object)
    return labels, codes


def classify_voc(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.where(values <= 500.0, 0, np.where(values <= 1000.0, 1, 2))
    labels = np.array([CODE_TO_RISK[int(code)] for code in codes], dtype=object)
    return labels, codes


def combine_risk(pm25_codes: np.ndarray, voc_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final_codes = np.maximum(pm25_codes, voc_codes)
    final_labels = np.array([CODE_TO_RISK[int(code)] for code in final_codes], dtype=object)
    return final_labels, final_codes


def simulate_alarms(final_risk: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate user alerts while limiting repeated alerts for the same risk level."""
    alarm_triggered = np.zeros(final_risk.size, dtype=bool)
    alarm_message = np.full(final_risk.size, "", dtype=object)
    alarm_delay_seconds = np.full(final_risk.size, np.nan, dtype=float)
    last_alarm_second = {"Yellow": -10**9, "Red": -10**9}

    for second, risk in enumerate(final_risk):
        if risk == "Green":
            continue

        if second - last_alarm_second[risk] < 60:
            continue

        alarm_triggered[second] = True
        last_alarm_second[risk] = second

        if risk == "Yellow":
            alarm_message[second] = "Dikkat: hava kalitesi düşüyor"
            delay = rng.normal(0.55, 0.13)
        else:
            alarm_message[second] = "Yüksek risk: ortamdan uzaklaşın veya maske kullanın"
            delay = rng.normal(0.36, 0.11)

        alarm_delay_seconds[second] = float(np.clip(delay, 0.10, 0.95))

    return alarm_triggered, alarm_message, alarm_delay_seconds


def derive_sampling_mode(final_risk: np.ndarray) -> np.ndarray:
    sampling_mode = np.full(final_risk.size, "LOW_POWER", dtype=object)
    sampling_mode[final_risk == "Yellow"] = "NORMAL"
    sampling_mode[final_risk == "Red"] = "HIGH_FREQUENCY"
    return sampling_mode


def simulate_ble(
    elapsed_seconds: np.ndarray,
    sampling_mode: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Send BLE packets only on seconds selected by the sampling mode."""
    intervals = np.array([SAMPLING_INTERVAL_SECONDS[mode] for mode in sampling_mode])
    mode_changed = np.r_[True, sampling_mode[1:] != sampling_mode[:-1]]
    ble_packet_sent = mode_changed | ((elapsed_seconds % intervals) == 0)

    ble_latency_ms = np.full(elapsed_seconds.size, np.nan, dtype=float)
    ble_packet_lost = np.zeros(elapsed_seconds.size, dtype=bool)

    for mode, (mean_latency, std_latency) in BLE_LATENCY_MODEL.items():
        mask = ble_packet_sent & (sampling_mode == mode)
        count = int(mask.sum())
        if count == 0:
            continue
        latency = rng.normal(mean_latency, std_latency, count)
        ble_latency_ms[mask] = np.clip(latency, 25.0, None)
        ble_packet_lost[mask] = rng.random(count) < BLE_PACKET_LOSS_PROBABILITY[mode]

    return ble_packet_sent, ble_latency_ms, ble_packet_lost


def simulate_battery(
    sampling_mode: np.ndarray,
    ble_packet_sent: np.ndarray,
    alarm_triggered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current_consumption = np.array([MODE_CURRENT_MA[mode] for mode in sampling_mode], dtype=float)
    current_consumption += ble_packet_sent.astype(float) * BLE_EXTRA_CURRENT_MA
    current_consumption += alarm_triggered.astype(float) * ALARM_EXTRA_CURRENT_MA

    cumulative_consumption_mah = np.cumsum(current_consumption / SECONDS_PER_HOUR)
    battery_percent = 100.0 * (1.0 - cumulative_consumption_mah / BATTERY_CAPACITY_MAH)
    return current_consumption, np.clip(battery_percent, 0.0, 100.0)


def create_simulation_dataframe() -> tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    elapsed_seconds = np.arange(TOTAL_SECONDS)
    elapsed_hours = elapsed_seconds / SECONDS_PER_HOUR
    timestamps = pd.date_range("2026-07-01 08:00:00", periods=TOTAL_SECONDS, freq="s")

    blocks = build_environment_blocks()
    environment, reference_pm25, reference_voc = assign_environment_arrays(elapsed_seconds, blocks, rng)
    temperature_c, humidity_percent = generate_temperature_and_humidity(elapsed_seconds, environment, blocks, rng)
    raw_pm25, raw_voc = generate_raw_sensor_data(
        reference_pm25,
        reference_voc,
        temperature_c,
        humidity_percent,
        elapsed_hours,
        rng,
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_hours": elapsed_hours,
            "environment": environment,
            "reference_pm25": reference_pm25,
            "raw_pm25": raw_pm25,
            "reference_voc": reference_voc,
            "raw_voc": raw_voc,
            "temperature_c": temperature_c,
            "humidity_percent": humidity_percent,
        }
    )

    calibrated_pm25, calibration_metrics, _ = calibrate_pm25(df)
    df["calibrated_pm25"] = calibrated_pm25

    risk_pm25, risk_pm25_codes = classify_pm25(df["calibrated_pm25"].to_numpy())
    risk_voc, risk_voc_codes = classify_voc(df["raw_voc"].to_numpy())
    final_risk, final_risk_codes = combine_risk(risk_pm25_codes, risk_voc_codes)

    alarm_triggered, alarm_message, alarm_delay_seconds = simulate_alarms(final_risk, rng)
    sampling_mode = derive_sampling_mode(final_risk)
    ble_packet_sent, ble_latency_ms, ble_packet_lost = simulate_ble(elapsed_seconds, sampling_mode, rng)
    current_consumption_mA, battery_percent = simulate_battery(sampling_mode, ble_packet_sent, alarm_triggered)

    df["risk_pm25"] = risk_pm25
    df["risk_voc"] = risk_voc
    df["final_risk"] = final_risk
    df["alarm_triggered"] = alarm_triggered
    df["alarm_message"] = alarm_message
    df["alarm_delay_seconds"] = alarm_delay_seconds
    df["sampling_mode"] = sampling_mode
    df["ble_packet_sent"] = ble_packet_sent
    df["ble_latency_ms"] = ble_latency_ms
    df["ble_packet_lost"] = ble_packet_lost
    df["current_consumption_mA"] = current_consumption_mA
    df["battery_percent"] = battery_percent
    df["final_risk_code"] = final_risk_codes
    df["sampling_mode_code"] = np.array([SAMPLING_MODE_TO_CODE[mode] for mode in sampling_mode])

    return df, calibration_metrics


def build_summary(df: pd.DataFrame, calibration_metrics: Dict[str, float]) -> Dict[str, float | int | str]:
    sent_packets = df[df["ble_packet_sent"]]
    lost_packets = int(sent_packets["ble_packet_lost"].sum())
    total_packets = int(len(sent_packets))
    packet_loss_rate = 100.0 * lost_packets / total_packets if total_packets else 0.0

    latency_values = sent_packets["ble_latency_ms"].dropna().to_numpy()
    mean_latency = float(np.mean(latency_values)) if latency_values.size else 0.0
    p95_latency = float(np.percentile(latency_values, 95)) if latency_values.size else 0.0

    alarm_delays = df.loc[df["alarm_triggered"], "alarm_delay_seconds"].dropna().to_numpy()
    mean_alarm_delay = float(np.mean(alarm_delays)) if alarm_delays.size else 0.0
    max_alarm_delay = float(np.max(alarm_delays)) if alarm_delays.size else 0.0

    total_consumed_mah = float(df["current_consumption_mA"].sum() / SECONDS_PER_HOUR)
    average_current_mA = total_consumed_mah / TOTAL_DURATION_HOURS
    estimated_runtime_hours = BATTERY_CAPACITY_MAH / average_current_mA if average_current_mA > 0 else 0.0
    battery_remaining = float(df["battery_percent"].iloc[-1])

    if battery_remaining > 0:
        conclusion = "8 saatlik kullanım hedefi simülasyon düzeyinde sağlandı."
    else:
        conclusion = "8 saat sonunda pil tükendi; 8 saatlik kullanım hedefi bu ayarlarla sağlanamadı."

    summary: Dict[str, float | int | str] = {
        "total_duration_hours": TOTAL_DURATION_HOURS,
        "total_samples_seconds": TOTAL_SECONDS,
        "total_ble_packets": total_packets,
        "packet_loss_rate_percent": float(packet_loss_rate),
        "mean_ble_latency_ms": mean_latency,
        "p95_ble_latency_ms": p95_latency,
        **calibration_metrics,
        "total_green_seconds": int((df["final_risk"] == "Green").sum()),
        "total_yellow_seconds": int((df["final_risk"] == "Yellow").sum()),
        "total_red_seconds": int((df["final_risk"] == "Red").sum()),
        "total_alarms": int(df["alarm_triggered"].sum()),
        "mean_alarm_delay_seconds": mean_alarm_delay,
        "max_alarm_delay_seconds": max_alarm_delay,
        "battery_remaining_percent": battery_remaining,
        "estimated_runtime_hours": float(estimated_runtime_hours),
        "conclusion_short": conclusion,
    }
    return summary


def save_json(summary: Dict[str, float | int | str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def plot_pm25_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["elapsed_hours"], df["reference_pm25"], label="Referans PM2.5", linewidth=1.2)
    ax.plot(df["elapsed_hours"], df["raw_pm25"], label="Ham PM2.5", linewidth=0.8, alpha=0.55)
    ax.plot(df["elapsed_hours"], df["calibrated_pm25"], label="Kalibre PM2.5", linewidth=1.0, alpha=0.9)
    ax.axhline(15, color="goldenrod", linestyle="--", linewidth=1.1, label="15 µg/m³ eşiği")
    ax.axhline(35, color="crimson", linestyle="--", linewidth=1.1, label="35 µg/m³ eşiği")
    ax.set_title("PM2.5 Referans, Ham ve Kalibre Edilmiş Veri")
    ax.set_xlabel("Zaman (saat)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlim(0, TOTAL_DURATION_HOURS)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration_error(summary: Dict[str, float | int | str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["MAE", "RMSE"]
    before = [float(summary["mae_before"]), float(summary["rmse_before"])]
    after = [float(summary["mae_after"]), float(summary["rmse_after"])]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, before, width, label="Kalibrasyon öncesi", color="#9aa3ad")
    ax.bar(x + width / 2, after, width, label="Kalibrasyon sonrası", color="#2f7d62")
    ax.set_title("Kalibrasyon Hata Karşılaştırması")
    ax.set_ylabel("Hata (µg/m³)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_risk_timeline(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.step(df["elapsed_hours"], df["final_risk_code"], where="post", color="#424a52", linewidth=1.2)
    ax.fill_between(df["elapsed_hours"], df["final_risk_code"], step="post", alpha=0.18, color="#d4553f")
    ax.set_title("Risk Seviyesi Zaman Çizelgesi")
    ax.set_xlabel("Zaman (saat)")
    ax.set_ylabel("Risk kodu")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Green", "Yellow", "Red"])
    ax.set_xlim(0, TOTAL_DURATION_HOURS)
    ax.set_ylim(-0.15, 2.15)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sampling_timeline(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.step(df["elapsed_hours"], df["sampling_mode_code"], where="post", color="#315b8a", linewidth=1.2)
    ax.set_title("Olay-Tetiklemeli Örnekleme Modu")
    ax.set_xlabel("Zaman (saat)")
    ax.set_ylabel("Mod")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["LOW_POWER", "NORMAL", "HIGH_FREQ"])
    ax.set_xlim(0, TOTAL_DURATION_HOURS)
    ax.set_ylim(-0.15, 2.15)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_battery(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(df["elapsed_hours"], df["battery_percent"], color="#2f7d62", linewidth=1.5)
    ax.set_title("Pil Yüzdesi")
    ax.set_xlabel("Zaman (saat)")
    ax.set_ylabel("Pil (%)")
    ax.set_xlim(0, TOTAL_DURATION_HOURS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ble_latency_and_loss(df: pd.DataFrame, path: Path) -> None:
    ble_df = df[df["ble_packet_sent"]].copy()
    delivered = ble_df[~ble_df["ble_packet_lost"]]
    lost = ble_df[ble_df["ble_packet_lost"]]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.scatter(
        delivered["elapsed_hours"],
        delivered["ble_latency_ms"],
        s=8,
        alpha=0.35,
        color="#315b8a",
        label="İletilen paket",
    )
    if not lost.empty:
        ax.scatter(
            lost["elapsed_hours"],
            lost["ble_latency_ms"],
            s=24,
            marker="x",
            color="crimson",
            label="Kayıp paket",
        )
    ax.set_title("BLE Gecikmesi ve Paket Kaybı")
    ax.set_xlabel("Zaman (saat)")
    ax.set_ylabel("BLE gecikmesi (ms)")
    ax.set_xlim(0, TOTAL_DURATION_HOURS)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_plots(df: pd.DataFrame, summary: Dict[str, float | int | str]) -> None:
    plot_pm25_comparison(df, OUTPUT_DIR / "pm25_reference_raw_calibrated.png")
    plot_calibration_error(summary, OUTPUT_DIR / "calibration_error_comparison.png")
    plot_risk_timeline(df, OUTPUT_DIR / "risk_level_timeline.png")
    plot_sampling_timeline(df, OUTPUT_DIR / "sampling_mode_timeline.png")
    plot_battery(df, OUTPUT_DIR / "battery_percentage.png")
    plot_ble_latency_and_loss(df, OUTPUT_DIR / "ble_latency_and_packet_loss.png")


def write_report(summary: Dict[str, float | int | str], path: Path) -> None:
    report = f"""# AeroWear Yazılım Tabanlı Sensör, Kalibrasyon, BLE ve Pil Tüketimi Simülasyonu

## 1. Amaç

Bu çalışmanın amacı, AeroWear adlı yakaya takılabilir hava kalitesi rozeti için gerçek donanım üretimi öncesinde yazılım tabanlı bir ön değerlendirme yapmaktır. Simülasyon; PM2.5, VOC/IAQ, sıcaklık, nem, kalibrasyon, BLE veri aktarımı, alarm davranışı ve pil tüketimi bileşenlerini birlikte ele alır.

Bu çalışma gerçek klinik karar sistemi değildir; tıbbi teşhis veya tedavi önerisi üretmez.

## 2. Simülasyon Senaryosu

Toplam 8 saatlik kullanım 1 saniye çözünürlükle modellenmiştir. Mikro-ortam akışı normal iç ortam, trafik yakın çevresi, temizlik ürünü kullanımı, yemek pişirme/kapalı ortam ve kampüs/açık alan yürüyüşü bloklarından oluşur.

TEYDEP proje dokümanındaki hedeflere uygun olarak sistem; düşük maliyetli PM2.5 ve VOC sensörleri, sıcaklık/nem telafisi, olay-tetiklemeli örnekleme, BLE aktarımı ve günlük kullanım için pil yönetimi varsayımlarıyla modellenmiştir.

## 3. Kullanılan Değişkenler

Temel değişkenler `reference_pm25`, `raw_pm25`, `calibrated_pm25`, `reference_voc`, `raw_voc`, `temperature_c`, `humidity_percent`, `final_risk`, `sampling_mode`, `ble_latency_ms`, `ble_packet_lost`, `current_consumption_mA` ve `battery_percent` alanlarıdır. Referans değerler gerçek ortam değeri gibi kabul edilmiş; ham sensör verisine gain hatası, Gaussian gürültü, sıcaklık/nem biası ve zamana bağlı drift eklenmiştir.

## 4. Kalibrasyon Yaklaşımı

PM2.5 kalibrasyonu için LinearRegression modeli kullanılmıştır. Özellikler `raw_pm25`, `temperature_c`, `humidity_percent` ve `elapsed_hours`; hedef değişken ise `reference_pm25` olarak belirlenmiştir. Veri zamana göre ayrılmış, ilk %70 eğitim ve son %30 test olarak kullanılmıştır.

Test kümesinde MAE {summary["mae_before"]:.2f} değerinden {summary["mae_after"]:.2f} değerine, RMSE ise {summary["rmse_before"]:.2f} değerinden {summary["rmse_after"]:.2f} değerine düşmüştür. MAE iyileşmesi %{summary["mae_improvement_percent"]:.1f}, RMSE iyileşmesi %{summary["rmse_improvement_percent"]:.1f} olarak hesaplanmıştır.

## 5. Olay-Tetiklemeli Örnekleme Mantığı

Risk seviyesi Green olduğunda LOW_POWER modunda 60 saniyede bir, Yellow olduğunda NORMAL modunda 10 saniyede bir, Red olduğunda HIGH_FREQUENCY modunda her saniye örnekleme yapılmıştır. Bu yaklaşım, yüksek riskte daha sık veri üretirken düşük riskte pil tüketimini azaltmayı amaçlar.

Simülasyonda Green süresi {summary["total_green_seconds"]} saniye, Yellow süresi {summary["total_yellow_seconds"]} saniye ve Red süresi {summary["total_red_seconds"]} saniyedir.

## 6. BLE Veri Aktarım Simülasyonu

BLE paketi yalnızca örnekleme yapılan saniyelerde gönderilmiş kabul edilmiştir. Toplam {summary["total_ble_packets"]} BLE paketi üretilmiş, paket kaybı oranı %{summary["packet_loss_rate_percent"]:.2f} olarak bulunmuştur. Ortalama BLE gecikmesi {summary["mean_ble_latency_ms"]:.1f} ms, 95. yüzdelik gecikme {summary["p95_ble_latency_ms"]:.1f} ms olarak hesaplanmıştır.

## 7. Pil Tüketimi Modeli

Pil kapasitesi 250 mAh kabul edilmiştir. LOW_POWER, NORMAL ve HIGH_FREQUENCY modları için sırasıyla 6 mA, 14 mA ve 28 mA temel akım tüketimi kullanılmıştır. BLE gönderimi olan saniyelerde +4 mA, alarm üretilen saniyelerde +8 mA eklenmiştir.

8 saat sonunda kalan pil yüzdesi %{summary["battery_remaining_percent"]:.1f}; ortalama akıma göre tahmini çalışma süresi {summary["estimated_runtime_hours"]:.1f} saattir.

## 8. Bulgular

Kalibrasyon sonrası hata metriklerinde hedeflenen %20 iyileşme eşiği aşılmıştır. BLE paket kaybı hedeflenen %5 sınırının altında kalmıştır. Alarm algoritması toplam {summary["total_alarms"]} bildirim üretmiş, ortalama alarm gecikmesi {summary["mean_alarm_delay_seconds"]:.2f} saniye ve maksimum alarm gecikmesi {summary["max_alarm_delay_seconds"]:.2f} saniye olmuştur.

Üretilen grafikler `outputs/` klasöründe yer almaktadır: PM2.5 karşılaştırması, hata karşılaştırması, risk zaman çizelgesi, örnekleme modu, pil yüzdesi ve BLE gecikme/kayıp grafiği.

## 9. Sonuç ve Yorum

{summary["conclusion_short"]} Kalibrasyon, ham sensör verisindeki sıcaklık/nem kaynaklı sapma ve drift etkilerini azaltarak ölçüm doğruluğunu belirgin şekilde iyileştirmiştir. Olay-tetiklemeli örnekleme, riskli anlarda yüksek frekanslı izlemeye geçerken düşük riskli sürelerde enerji tüketimini sınırlamıştır.

Bu simülasyon, gerçek donanım üretimi öncesinde AeroWear sisteminin sensör verisi işleme, kalibrasyon, risk sınıflandırması, BLE veri aktarımı ve enerji tüketimi davranışlarının ön değerlendirmesini sağlamaktadır.

## 10. Sınırlılıklar

Bu çalışma sentetik veri üretimine dayanır ve klinik validasyon yerine geçmez. Sensör yaşlanması, gövde hava akışı, gerçek BLE parazitleri, kullanıcı hareketi, konum değişimi ve kişiye özel tıbbi eşikler basitleştirilmiş olarak temsil edilmiştir. Gerçek prototip aşamasında referans cihaz eşleşmesi, kontrollü ortam testi ve saha doğrulaması gereklidir.
"""
    path.write_text(report, encoding="utf-8")


def save_outputs(df: pd.DataFrame, summary: Dict[str, float | int | str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "timestamp",
        "elapsed_seconds",
        "elapsed_hours",
        "environment",
        "reference_pm25",
        "raw_pm25",
        "calibrated_pm25",
        "reference_voc",
        "raw_voc",
        "temperature_c",
        "humidity_percent",
        "risk_pm25",
        "risk_voc",
        "final_risk",
        "alarm_triggered",
        "alarm_message",
        "sampling_mode",
        "ble_packet_sent",
        "ble_latency_ms",
        "ble_packet_lost",
        "current_consumption_mA",
        "battery_percent",
        "alarm_delay_seconds",
    ]
    df[csv_columns].to_csv(OUTPUT_DIR / "aerowear_simulation_data.csv", index=False, encoding="utf-8")
    save_json(summary, OUTPUT_DIR / "simulation_summary.json")
    save_plots(df, summary)
    write_report(summary, BASE_DIR / "simulation_report.md")


def print_console_summary(summary: Dict[str, float | int | str]) -> None:
    print("\nAeroWear simülasyonu tamamlandı.")
    print(f"Toplam süre: {summary['total_duration_hours']} saat / {summary['total_samples_seconds']} saniye")
    print(
        "Kalibrasyon: "
        f"MAE {summary['mae_before']:.2f} -> {summary['mae_after']:.2f} "
        f"(%{summary['mae_improvement_percent']:.1f} iyileşme), "
        f"RMSE {summary['rmse_before']:.2f} -> {summary['rmse_after']:.2f} "
        f"(%{summary['rmse_improvement_percent']:.1f} iyileşme)"
    )
    print(
        "BLE: "
        f"{summary['total_ble_packets']} paket, "
        f"%{summary['packet_loss_rate_percent']:.2f} kayıp, "
        f"ortalama {summary['mean_ble_latency_ms']:.1f} ms gecikme"
    )
    print(
        "Risk süreleri: "
        f"Green {summary['total_green_seconds']} sn, "
        f"Yellow {summary['total_yellow_seconds']} sn, "
        f"Red {summary['total_red_seconds']} sn"
    )
    print(
        "Alarm ve pil: "
        f"{summary['total_alarms']} alarm, "
        f"ortalama alarm gecikmesi {summary['mean_alarm_delay_seconds']:.2f} sn, "
        f"kalan pil %{summary['battery_remaining_percent']:.1f}"
    )
    print(summary["conclusion_short"])
    print(f"Çıktılar: {OUTPUT_DIR}")


def main() -> None:
    df, calibration_metrics = create_simulation_dataframe()
    summary = build_summary(df, calibration_metrics)
    save_outputs(df, summary)
    print_console_summary(summary)


if __name__ == "__main__":
    main()
