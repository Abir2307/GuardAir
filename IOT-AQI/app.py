import streamlit as st
import os
import sys

# Configure environment logs before loading major frameworks
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", message=".*Skipping variable loading for optimizer.*")
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import time
import hashlib
from io import BytesIO

# Establish absolute basePath configuration for server-side subdirectories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_relative_path(filename):
    return os.path.join(BASE_DIR, filename)

EXCEL_PATH = get_relative_path("Excel_IOT BASED REAL TIME AIR QUALITY MONITORING SYSTEM.xlsx")
EXCEL_FEATURE_COLUMNS = ["Temperature (°C)", "Humidity (%)", "PM2.5", "MQ135 (Gas)", "MQ7 (CO)"]
EXCEL_TIMESTAMP_COLUMN = "Timestamp"
FEATURE_NAMES = ["temp", "humidity", "pm25", "mq135", "mq7"]
CONTRIB_BACKGROUND_SAMPLES = 10

# Load deserialized pipelines and binary frameworks using dynamic paths
scaler = joblib.load(get_relative_path("scaler.pkl"))
if_model = joblib.load(get_relative_path("isolation_forest_model.pkl"))
ae_model = tf.keras.models.load_model(get_relative_path("autoencoder.keras"))

# Load runtime distribution vectors
ae_dist_path = get_relative_path("ae_error_dist.npy")
if_dist_path = get_relative_path("if_score_dist.npy")
ae_dist = np.load(ae_dist_path) if os.path.exists(ae_dist_path) else None
if_dist = np.load(if_dist_path) if os.path.exists(if_dist_path) else None

# Session State Architecture initialization
if "prev_ae" not in st.session_state:
    st.session_state["prev_ae"] = None

if "alerts" not in st.session_state:
    st.session_state["alerts"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

if "last_contrib" not in st.session_state:
    st.session_state["last_contrib"] = None

if "paused" not in st.session_state:
    st.session_state["paused"] = False

if "stream_cursor" not in st.session_state:
    st.session_state["stream_cursor"] = 0

if "latest_sample" not in st.session_state:
    st.session_state["latest_sample"] = None

if "source_status" not in st.session_state:
    st.session_state["source_status"] = "Waiting for Excel data"

if "active_sheet" not in st.session_state:
    st.session_state["active_sheet"] = None

if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Device workbook"

if "uploaded_excel_bytes" not in st.session_state:
    st.session_state["uploaded_excel_bytes"] = None

if "uploaded_excel_name" not in st.session_state:
    st.session_state["uploaded_excel_name"] = None

if "last_contrib_index" not in st.session_state:
    st.session_state["last_contrib_index"] = -1

if "smoothed_health" not in st.session_state:
    st.session_state["smoothed_health"] = 50.0


def _sanitize_health_thresholds(raw_thresholds):
    if not isinstance(raw_thresholds, dict):
        return None
    try:
        critical = float(raw_thresholds.get("critical"))
        warning = float(raw_thresholds.get("warning"))
        healthy = float(raw_thresholds.get("healthy"))
    except (TypeError, ValueError):
        return None

    if not (0 <= critical <= 100 and 0 <= warning <= 100 and 0 <= healthy <= 100):
        return None
    if not (critical <= warning < healthy):
        return None
    return {
        "critical": critical,
        "warning": warning,
        "healthy": healthy,
    }


def _load_health_thresholds():
    threshold_file_candidates = [get_relative_path("health_threshold.npy"), get_relative_path("health_thresholds.npy")]
    for threshold_file in threshold_file_candidates:
        if os.path.exists(threshold_file):
            try:
                loaded = np.load(threshold_file, allow_pickle=True).item()
                sanitized = _sanitize_health_thresholds(loaded)
                if sanitized is not None:
                    return sanitized, threshold_file
            except Exception:
                pass
    return (
        {
            "critical": 0.0,
            "warning": 40.0,
            "healthy": 70.0,
        },
        "defaults",
    )


HEALTH_THRESHOLDS, HEALTH_THRESHOLD_SOURCE = _load_health_thresholds()
CRITICAL_MIN = HEALTH_THRESHOLDS["critical"]
WARNING_MIN = HEALTH_THRESHOLDS["warning"]
HEALTHY_MIN = HEALTH_THRESHOLDS["healthy"]


def classify_alert_type(health, confidence_score):
    if confidence_score < 50:
        return "UNCERTAIN"
    if health >= HEALTHY_MIN:
        return "HEALTHY"
    if health >= WARNING_MIN:
        return "WARNING"

    recent_health = st.session_state.get("history", [])[-4:]
    sustained_low = len(recent_health) == 4 and all(h < WARNING_MIN for h in recent_health)
    deep_critical = health < CRITICAL_MIN

    if deep_critical or sustained_low:
        return "CRITICAL"
    return "WARNING"


def health_to_equal_band_value(health):
    health_clamped = float(np.clip(health, 0.0, 100.0))
    lower_span = max(CRITICAL_MIN, 1e-6)
    middle_span = max(WARNING_MIN - CRITICAL_MIN, 1e-6)
    upper_span = max(100.0 - WARNING_MIN, 1e-6)

    if health_clamped < CRITICAL_MIN:
        return (health_clamped / lower_span) * 33.333333
    if health_clamped < WARNING_MIN:
        return 33.333333 + ((health_clamped - CRITICAL_MIN) / middle_span) * 33.333333
    return 66.666666 + ((health_clamped - WARNING_MIN) / upper_span) * 33.333334


def normalize_alert_type(alert_type):
    if alert_type == "DEGRADED":
        return "WARNING"
    return alert_type


def status_style(value):
    status = str(value).upper()
    if status == "WARNING":
        return "color: #f59e0b; font-weight: 700"
    if status in {"HEALTHY", "NORMAL"}:
        return "color: #16a34a; font-weight: 700"
    if status == "CRITICAL":
        return "color: #dc2626; font-weight: 700"
    if status == "UNCERTAIN":
        return "color: #6b7280; font-weight: 700"
    return ""


@st.cache_data(show_spinner=False)
def _cached_sheet_names_device(path, mtime):
    del mtime
    workbook = pd.ExcelFile(path)
    return workbook.sheet_names


@st.cache_data(show_spinner=False)
def _cached_sheet_names_upload(upload_digest, uploaded_bytes):
    del upload_digest
    workbook = pd.ExcelFile(BytesIO(uploaded_bytes))
    return workbook.sheet_names


@st.cache_data(show_spinner=False)
def _cached_rows_device(path, sheet_name, mtime):
    del mtime
    frame = pd.read_excel(path, sheet_name=sheet_name)
    return frame


@st.cache_data(show_spinner=False)
def _cached_rows_upload(upload_digest, uploaded_bytes, sheet_name):
    del upload_digest
    frame = pd.read_excel(BytesIO(uploaded_bytes), sheet_name=sheet_name)
    return frame


def _bytes_digest(uploaded_bytes):
    if uploaded_bytes is None:
        return None
    return hashlib.md5(uploaded_bytes).hexdigest()


def load_excel_sheet_names(source_mode, uploaded_bytes=None):
    if source_mode == "Upload Excel file":
        if uploaded_bytes is None:
            return [], "Upload an .xlsx file to begin streaming"
        try:
            digest = _bytes_digest(uploaded_bytes)
            sheet_names = _cached_sheet_names_upload(digest, uploaded_bytes)
        except Exception as exc:
            return [], f"Unable to inspect uploaded Excel file yet: {exc}"
        return sheet_names, f"Loaded {len(sheet_names)} sheet(s) from upload"

    if not os.path.exists(EXCEL_PATH):
        return [], f"Excel file not found: {EXCEL_PATH}"
    try:
        mtime = os.path.getmtime(EXCEL_PATH)
        sheet_names = _cached_sheet_names_device(EXCEL_PATH, mtime)
    except Exception as exc:
        return [], f"Unable to inspect Excel file yet: {exc}"
    return sheet_names, f"Loaded {len(sheet_names)} sheet(s)"


def load_excel_rows(sheet_name, source_mode, uploaded_bytes=None):
    if source_mode == "Upload Excel file":
        if uploaded_bytes is None:
            return None, "Upload an .xlsx file to begin streaming"
        try:
            digest = _bytes_digest(uploaded_bytes)
            frame = _cached_rows_upload(digest, uploaded_bytes, sheet_name)
        except Exception as exc:
            return None, f"Unable to read uploaded Excel sheet '{sheet_name}' yet: {exc}"

        missing_columns = [column for column in EXCEL_FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"

        cleaned = frame.copy()
        cleaned = cleaned[EXCEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        cleaned = cleaned.dropna(subset=EXCEL_FEATURE_COLUMNS)
        if cleaned.empty:
            return None, "No usable sensor rows found in uploaded Excel"
        return cleaned.reset_index(drop=True), f"Loaded {len(cleaned)} rows from uploaded Excel"

    if not os.path.exists(EXCEL_PATH):
        return None, f"Excel file not found: {EXCEL_PATH}"
    try:
        mtime = os.path.getmtime(EXCEL_PATH)
        frame = _cached_rows_device(EXCEL_PATH, sheet_name, mtime)
    except Exception as exc:
        return None, f"Unable to read Excel sheet '{sheet_name}' yet: {exc}"

    missing_columns = [column for column in EXCEL_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        return None, f"Missing required columns: {', '.join(missing_columns)}"

    cleaned = frame.copy()
    cleaned = cleaned[EXCEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(subset=EXCEL_FEATURE_COLUMNS)
    if EXCEL_TIMESTAMP_COLUMN in frame.columns:
        cleaned[EXCEL_TIMESTAMP_COLUMN] = pd.to_datetime(frame[EXCEL_TIMESTAMP_COLUMN], errors="coerce")

    if cleaned.empty:
        return None, "No usable sensor rows found in Excel"
    return cleaned.reset_index(drop=True), f"Loaded {len(cleaned)} rows from Excel"


def row_to_sample(row):
    return pd.DataFrame([[row[column] for column in EXCEL_FEATURE_COLUMNS]], columns=EXCEL_FEATURE_COLUMNS)


def row_timestamp(row):
    if hasattr(row, 'index') and EXCEL_TIMESTAMP_COLUMN in row.index:
        timestamp_value = pd.to_datetime(row[EXCEL_TIMESTAMP_COLUMN], errors="coerce")
        if pd.notna(timestamp_value):
            return timestamp_value
    return pd.Timestamp.now()


def compute_stability(current_ae):
    prev = st.session_state["prev_ae"]
    if prev is None:
        stability = 1
    else:
        threshold = np.percentile([prev, current_ae], 75)
        stability = 1 if abs(current_ae - prev) < threshold else 0
    st.session_state["prev_ae"] = current_ae
    return stability


def rule_based(x):
    if isinstance(x, pd.DataFrame):
        temp, hum, pm, mq135, mq7 = x.iloc[0][EXCEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    else:
        temp, hum, pm, mq135, mq7 = x[0]

    hard_fault = (pm >= 4000 or mq135 >= 4000 or mq7 >= 4000)
    spike = ((1000 < pm < 4000) or (2000 < mq135 < 4000) or (100 < mq7 < 4000))

    if hard_fault:
        return 1.0
    if spike:
        return 0.5
    return 0.0


def ae_error(x_scaled):
    x_array = np.asarray(x_scaled, dtype=float)
    recon = ae_model.predict(x_array, verbose=0)
    return np.mean((x_array - recon) ** 2)


def ae_error_batch(x_scaled):
    x_array = np.asarray(x_scaled, dtype=float)
    recon = ae_model.predict(x_array, verbose=0)
    return np.mean((x_array - recon) ** 2, axis=1)


def feature_contribution(sample, X_reference=None):
    scaled_sample = scaler.transform(sample)
    base_pred = ae_error(scaled_sample)
    contributions = {}

    if X_reference is None or len(X_reference) == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    reference_values = X_reference[EXCEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    sample_values = sample[EXCEL_FEATURE_COLUMNS].to_numpy(dtype=float)

    for i, name in enumerate(FEATURE_NAMES):
        sample_count = min(CONTRIB_BACKGROUND_SAMPLES, len(reference_values))
        random_idx = np.random.choice(len(reference_values), sample_count, replace=len(reference_values) < sample_count)
        background_values = reference_values[random_idx]
        perturbed = np.repeat(sample_values, sample_count, axis=0)
        perturbed[:, i] = background_values[:, i]
        scaled_perturbed = scaler.transform(perturbed)
        perturbed_preds = ae_error_batch(scaled_perturbed)
        contributions[name] = float(base_pred - np.mean(perturbed_preds))

    total = sum(abs(v) for v in contributions.values()) + 1e-8
    return {key: value / total for key, value in contributions.items()}


def predict(sample):
    scaled = scaler.transform(sample)
    rule = rule_based(sample)
    
    ml_raw = -if_model.score_samples(sample)[0]
    p_if = (if_dist <= ml_raw).mean() if if_dist is not None else (1 / (1 + np.exp(-ml_raw)))
    if_risk = 1 - p_if

    ae = ae_error(scaled)
    ae_risk = np.clip(ae / 0.03, 0, 1)

    prob = 1 - (1 - rule) * (1 - if_risk) * (1 - ae_risk)
    stability = compute_stability(ae)

    if stability == 0:
        prob *= 0.85

    return {
        "rule": rule,
        "ml": p_if,
        "ml_raw": ml_raw,
        "ae": ae,
        "score": prob,
        "stability": stability,
    }


def confidence(rule, ml, ae):
    signals = [rule, ml, ae]
    return (1 - np.std(signals)) * 100


def health_percentage(score):
    return max(0, min(100, (1 - score) * 100))


def smooth_health(new_health):
    st.session_state["smoothed_health"] = new_health
    return new_health


def health_stats(window_size):
    history = st.session_state.get("history", [])
    if not history:
        return 0.0, 0.0, 0.0, 0.0
    avg_health = float(np.mean(history))
    min_health = float(np.min(history))
    max_health = float(np.max(history))
    moving_avg = float(np.mean(history[-window_size:]))
    return avg_health, min_health, max_health, moving_avg


if_thresh_path = get_relative_path("if_thresholds.npy")
ae_thresh_path = get_relative_path("ae_thresholds.npy")

if_thresholds = np.load(if_thresh_path, allow_pickle=True).item() if os.path.exists(if_thresh_path) else {"critical": 0.5, "suspicious": 0.3, "slightly_unusual": 0.1}
ae_thresholds = np.load(ae_thresh_path, allow_pickle=True).item() if os.path.exists(ae_thresh_path) else {"critical": 0.05, "suspicious": 0.02, "slight": 0.01}


def classify_if(score):
    if score >= if_thresholds["critical"]: return "strong anomaly"
    if score >= if_thresholds["suspicious"]: return "suspicious pattern"
    if score >= if_thresholds["slightly_unusual"]: return "slightly unusual"
    return "normal"


def classify_ae(error):
    if error >= ae_thresholds["critical"]: return "critical anomaly"
    if error >= ae_thresholds["suspicious"]: return "suspicious anomaly"
    if error >= ae_thresholds["slight"]: return "slight anomaly"
    return "normal"


def explain(rule, ml, ae, health, alert_type):
    reasons = []
    if rule == 1.0: reasons.append("Hardware fault detected")
    elif rule == 0.5: reasons.append("Spike anomaly detected")

    ml_class = classify_if(st.session_state.get("ml_raw", 0))
    if ml_class in ["strong anomaly", "suspicious pattern", "slightly unusual"]:
        reasons.append(f"Isolation Forest: {ml_class}")
    
    ae_class = classify_ae(st.session_state.get("ae", 0))
    if ae_class in ["critical anomaly", "suspicious anomaly", "slight anomaly"]:
        reasons.append(f"Autoencoder: {ae_class}")

    if not reasons:
        recent_health = st.session_state.get("history", [])[-4:]
        sustained_low = len(recent_health) == 4 and all(h < WARNING_MIN for h in recent_health)
        if alert_type == "CRITICAL":
            if health < CRITICAL_MIN:
                reasons.append(f"Health dropped into deep critical zone ({health:.2f}% < {CRITICAL_MIN:.2f}%)")
            elif sustained_low:
                reasons.append(f"Health stayed below warning threshold for 4 consecutive states (< {WARNING_MIN:.2f}%)")
            else:
                reasons.append("Critical state triggered by sustained low health trend")
        elif alert_type == "WARNING":
            if health < WARNING_MIN:
                reasons.append(f"Health dipped below warning threshold ({health:.2f}% < {WARNING_MIN:.2f}%), waiting for persistence before CRITICAL")
            else:
                reasons.append(f"Combined health is in warning band ({health:.2f}%), between {WARNING_MIN:.2f}% and {HEALTHY_MIN:.2f}%")
        elif alert_type == "UNCERTAIN":
            reasons.append("Confidence is low, so the reading is marked uncertain")
        else:
            reasons.append("No strong anomaly signal from rule, Isolation Forest, or autoencoder")
    return reasons


st.session_state["alerts"] = [
    {**alert, "type": normalize_alert_type(alert.get("type"))}
    for alert in st.session_state["alerts"]
]

st.set_page_config(page_title="IoT Anomaly Dashboard", layout="wide")
st.title("Real-Time IoT Fault Detection System")

with st.sidebar:
    st.header("Display Controls")
    source_mode = st.radio("Data Source", ["Device workbook", "Upload Excel file"], index=0)
    st.session_state["data_source"] = source_mode

    uploaded_file = None
    if source_mode == "Upload Excel file":
        uploaded_file = st.file_uploader("Upload .xlsx", type=["xlsx"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            if (st.session_state.get("uploaded_excel_name") != uploaded_file.name or 
                st.session_state.get("uploaded_excel_bytes") != uploaded_bytes):
                st.session_state["uploaded_excel_name"] = uploaded_file.name
                st.session_state["uploaded_excel_bytes"] = uploaded_bytes
                st.session_state["active_sheet"] = None
                st.session_state["selected_sheet"] = None
                st.session_state["stream_cursor"] = 0
                st.session_state["latest_sample"] = None
                st.session_state["last_contrib"] = None
                st.session_state["last_contrib_index"] = -1
                st.session_state["prev_ae"] = None
                st.session_state["history"] = []
                st.session_state["smoothed_health"] = 50.0
                st.session_state["source_status"] = f"Uploaded {uploaded_file.name}"
        else:
            st.session_state["uploaded_excel_bytes"] = None
            st.session_state["uploaded_excel_name"] = None

    auto_refresh = st.toggle("Auto Refresh", value=True)
    refresh_seconds = st.slider("Refresh Interval (sec)", 1, 100, 5)
    rows_per_refresh = st.slider("Rows Processed / Refresh", 1, 10, 1)
    history_window = st.slider("History Window", 30, 300, 60)
    moving_avg_window = st.slider("Moving Avg Window", 5, 100, 2)
    
    sheet_names, sheet_status = load_excel_sheet_names(source_mode, st.session_state.get("uploaded_excel_bytes"))
    if sheet_names:
        active_sheet = st.session_state.get("active_sheet")
        if active_sheet not in sheet_names:
            active_sheet = sheet_names[0]
        st.selectbox("Excel Sheet", sheet_names, index=sheet_names.index(active_sheet), key="selected_sheet")
        selected_sheet = st.session_state["selected_sheet"]
        st.caption(f"Active Excel sheet: {selected_sheet}")
        if st.session_state.get("active_sheet") != selected_sheet:
            st.session_state["active_sheet"] = selected_sheet
            st.session_state["stream_cursor"] = 0
            st.session_state["latest_sample"] = None
            st.session_state["last_contrib"] = None
            st.session_state["last_contrib_index"] = -1
            st.session_state["prev_ae"] = None
            st.session_state["history"] = []
            st.session_state["smoothed_health"] = 50.0
            st.session_state["source_status"] = f"Sheet changed to {selected_sheet}"
    else:
        st.session_state["selected_sheet"] = None
        st.session_state["active_sheet"] = None
        st.caption(sheet_status)
    
    st.divider()
    pause_notice = st.empty()
    if st.button("⏸ Pause" if not st.session_state["paused"] else "▶ Resume", width="stretch"):
        st.session_state["paused"] = not st.session_state["paused"]
    if st.session_state["paused"]:
        pause_notice.info("Dashboard paused - values frozen")
    else:
        pause_notice.empty()

placeholder = st.empty()

selected_sheet = st.session_state.get("active_sheet")
excel_rows, excel_status = (
    load_excel_rows(selected_sheet, st.session_state["data_source"], st.session_state.get("uploaded_excel_bytes"))
    if selected_sheet else (None, "No Excel sheet selected")
)
st.session_state["source_status"] = excel_status

if excel_rows is not None and st.session_state["stream_cursor"] > len(excel_rows):
    st.session_state["stream_cursor"] = 0

if excel_rows is not None:
    if st.session_state["stream_cursor"] >= len(excel_rows):
        st.session_state["source_status"] = f"Excel loaded with {len(excel_rows)} rows. Waiting for new rows."
    else:
        next_end = min(st.session_state["stream_cursor"] + rows_per_refresh, len(excel_rows))
        st.session_state["source_status"] = f"Streaming rows {st.session_state['stream_cursor'] + 1} to {next_end} of {len(excel_rows)}"

i = st.session_state.get("counter", 0)
no_live_sample = False

if not st.session_state["paused"]:
    if excel_rows is not None and not excel_rows.empty:
        start_index = st.session_state["stream_cursor"]

        if start_index >= len(excel_rows):
            sample = st.session_state.get("latest_sample")
            if sample is None:
                sample = row_to_sample(excel_rows.iloc[-1])

            rule = st.session_state.get("rule", 0.0)
            ml = st.session_state.get("ml", 0.0)
            ml_raw = st.session_state.get("ml_raw", 0.0)
            ae = st.session_state.get("ae", 0.0)
            final_score = st.session_state.get("final_score", 0.0)
            stability = st.session_state.get("stability", 1)
            health = st.session_state.get("smoothed_health", 50.0)
            conf = confidence(rule, ml, ae)
            alert_type = "NORMAL"
            cyber_status = "Normal"
        else:
            batch_end = min(start_index + rows_per_refresh, len(excel_rows))
            sample = None

            for row_index in range(start_index, batch_end):
                latest_row = excel_rows.iloc[row_index]
                sample = row_to_sample(latest_row)
                pm25 = sample["PM2.5"].iloc[0]
                mq135 = sample["MQ135 (Gas)"].iloc[0]
                mq7 = sample["MQ7 (CO)"].iloc[0]

                saturation_attack = (pm25 >= 4095 or mq135 >= 4095 or mq7 >= 4095)
                spoofing_attack = (pm25 > 1000 and mq135 < 300)
                cyber = int(saturation_attack or spoofing_attack)
                cyber_status = "Potential Attack" if cyber == 1 else "Normal"

                out = predict(sample)
                rule = out["rule"]
                ml = out["ml"]
                ml_raw = out["ml_raw"]
                ae = out["ae"]
                final_score = out["score"]
                stability = out["stability"]

                raw_health = health_percentage(final_score)
                health = smooth_health(raw_health)
                conf = confidence(rule, ml, ae)

                st.session_state["history"].append(health)
                alert_type = classify_alert_type(health, conf)

                st.session_state["alerts"].append({
                    "time": row_timestamp(latest_row),
                    "score": final_score,
                    "trust_score": conf,
                    "cyber": cyber,
                    "health": health,
                    "cyber_status": cyber_status,
                    "type": alert_type,
                })
                st.session_state["alerts"] = st.session_state["alerts"][-500:]

                st.session_state["rule"] = rule
                st.session_state["ml"] = ml
                st.session_state["ml_raw"] = ml_raw
                st.session_state["ae"] = ae
                st.session_state["final_score"] = final_score
                st.session_state["stability"] = stability
                st.session_state["latest_sample"] = sample

            st.session_state["stream_cursor"] = batch_end
            st.session_state["counter"] = i + max(1, batch_end - start_index)
            i = st.session_state["counter"]

            if sample is not None:
                st.session_state["last_contrib"] = feature_contribution(sample, excel_rows)
                st.session_state["last_contrib_index"] = batch_end - 1
    else:
        sample = st.session_state.get("latest_sample")
        if sample is None:
            no_live_sample = True
            rule = st.session_state.get("rule", 0.0)
            ml = st.session_state.get("ml", 0.0)
            ml_raw = st.session_state.get("ml_raw", 0.0)
            ae = st.session_state.get("ae", 0.0)
            final_score = st.session_state.get("final_score", 0.0)
            stability = st.session_state.get("stability", 1)
            health = st.session_state.get("smoothed_health", 50.0)
            conf = confidence(rule, ml, ae)
            alert_type = "NORMAL"
            cyber_status = "Normal"
        else:
            pm25 = sample["PM2.5"].iloc[0]
            mq135 = sample["MQ135 (Gas)"].iloc[0]
            mq7 = sample["MQ7 (CO)"].iloc[0]
            
            saturation_attack = (pm25 >= 4095 or mq135 >= 4095 or mq7 >= 4095)
            spoofing_attack = (pm25 > 1000 and mq135 < 300)
            cyber = int(saturation_attack or spoofing_attack)
            cyber_status = "Potential Attack" if cyber == 1 else "Normal"

            out = predict(sample)
            rule = out["rule"]
            ml = out["ml"]
            ml_raw = out["ml_raw"]
            ae = out["ae"]
            final_score = out["score"]
            stability = out["stability"]

            raw_health = health_percentage(final_score)
            health = smooth_health(raw_health)
            conf = confidence(rule, ml, ae)

            st.session_state["history"].append(health)
            alert_type = classify_alert_type(health, conf)

            st.session_state["alerts"].append({
                "time": row_timestamp(sample.iloc[0]),
                "score": final_score,
                "trust_score": conf,
                "health": health,
                "cyber_status": cyber_status,
                "type": alert_type,
                "cyber": cyber
            })
            st.session_state["alerts"] = st.session_state["alerts"][-500:]

            st.session_state["rule"] = rule
            st.session_state["ml"] = ml
            st.session_state["ml_raw"] = ml_raw
            st.session_state["ae"] = ae
            st.session_state["final_score"] = final_score
            st.session_state["stability"] = stability
else:
    rule = st.session_state.get("rule", 0.0)
    ml = st.session_state.get("ml", 0.0)
    ml_raw = st.session_state.get("ml_raw", 0.0)
    ae = st.session_state.get("ae", 0.0)
    final_score = st.session_state.get("final_score", 0.0)
    stability = st.session_state.get("stability", 1)
    health = st.session_state.get("smoothed_health", 50.0)
    conf = confidence(rule, ml, ae)
    alert_type = "NORMAL"

with placeholder.container():
    st.subheader("System Health")
    st.caption(st.session_state.get("source_status", "Excel source unavailable"))
    if no_live_sample:
        st.warning("No live sample available.")
    if selected_sheet:
        st.caption(f"Current tab: {selected_sheet}")
    if excel_rows is not None and not excel_rows.empty:
        st.caption(f"Processed rows: {st.session_state.get('stream_cursor', 0)} / {len(excel_rows)}")

    if excel_rows is not None and not excel_rows.empty:
        current_idx = min(max(st.session_state.get("stream_cursor", 1) - 1, 0), len(excel_rows) - 1)
        st.subheader("Live Input Row")
        current_row = excel_rows.iloc[current_idx]
        if EXCEL_TIMESTAMP_COLUMN in current_row.index:
            st.caption(f"Source timestamp: {current_row[EXCEL_TIMESTAMP_COLUMN]}")
        st.dataframe(excel_rows.iloc[[current_idx]], width="stretch", height=110)

    avg_health, min_health, max_health, moving_avg = health_stats(moving_avg_window)
    gauge_col, stats_col = st.columns([3, 2])

    with gauge_col:
        gauge_value = health_to_equal_band_value(health)
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=gauge_value,
            title={"text": "Health %"},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "rgba(0,0,0,0)"}},
                "bar": {"color": "blue"},
                "steps": [
                    {"range": [0, 33.333333], "color": "red"},
                    {"range": [33.333333, 66.666666], "color": "yellow"},
                    {"range": [66.666666, 100], "color": "green"},
                ],
            },
        ))
        fig.add_annotation(
            x=0.5, y=0.13,
            text=f"{health:.2f}%",
            showarrow=False,
            font={"size": 24},
        )
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, width="stretch", key="gauge")

    with stats_col:
        st.subheader("Health Stats")
        st.metric("Average", round(avg_health, 2))
        st.metric("Minimum", round(min_health, 2))
        st.metric("Maximum", round(max_health, 2))
        st.metric(f"Moving Avg ({moving_avg_window})", round(moving_avg, 2))

    if alert_type == "CRITICAL":
        st.error("CRITICAL SYSTEM FAILURE")
    elif alert_type in {"WARNING", "UNCERTAIN"}:
        st.warning("SYSTEM WARNING")
    else:
        st.success("SYSTEM HEALTHY")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rule-Based", int(rule))
    col2.metric("Isolation Raw", round(ml_raw, 6))
    col3.metric("AE Error", round(ae, 6))
    col4, col5 = st.columns(2)
    col4.metric("Confidence %", round(conf, 2))
    col5.metric("Stability", stability)

    st.subheader("Explanation")
    reasons = explain(rule, ml, ae, health, alert_type)
    for reason in reasons:
        st.write("-", reason)

    st.subheader("Trend")
    trend_fig = go.Figure()
    trend_fig.add_scatter(y=st.session_state["history"], mode="lines", name="Health")
    trend_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, max(len(st.session_state["history"]) - 1, 1)], fixedrange=True),
        yaxis=dict(range=[0, 100], fixedrange=True),
        height=260,
    )
    st.plotly_chart(trend_fig, width="stretch", key="trend")

    st.subheader("Live Status Logs")
    alert_df = pd.DataFrame(st.session_state["alerts"])

    if not alert_df.empty:
        display_alert_df = alert_df.rename(columns={
            "time": "Timestamp",
            "score": "Anomaly Score",
            "trust_score": "Trust Score",
            "health": "Health",
            "cyber_status": "Cyber Status",
            "type": "System Status"
        })
        required_cols = ["Timestamp", "Anomaly Score", "Trust Score", "Health", "Cyber Status", "System Status"]
        display_alert_df = display_alert_df[required_cols]
        styled_alert_df = display_alert_df.tail(25).style.map(status_style, subset=["System Status"])
        st.dataframe(styled_alert_df, width="stretch", height=260)
    else:
        display_alert_df = pd.DataFrame(columns=["Timestamp", "Anomaly Score", "Trust Score", "Health", "Cyber Status", "System Status"])
        st.dataframe(display_alert_df, width="stretch", height=260)

    st.subheader("Feature Contribution")
    current_index = max(st.session_state.get("stream_cursor", 1) - 1, 0)
    should_update_contrib = (
        not st.session_state["paused"]
        and excel_rows is not None
        and not excel_rows.empty
        and sample is not None
        and current_index != st.session_state.get("last_contrib_index", -1)
    )

    if (st.session_state["last_contrib"] is None and sample is not None and excel_rows is not None and not excel_rows.empty) or should_update_contrib:
        st.session_state["last_contrib"] = feature_contribution(sample, excel_rows)
        st.session_state["last_contrib_index"] = current_index
    
    contrib = st.session_state.get("last_contrib") or {}
    feature_order = ["temp", "humidity", "pm25", "mq135", "mq7"]

    df_contrib = pd.DataFrame({
        "Feature": feature_order,
        "Impact": [contrib.get(f, 0.0) for f in feature_order]
    })

    colors = ["red" if v > 0 else "green" for v in df_contrib["Impact"]]
    fig = go.Figure(go.Bar(
        x=df_contrib["Feature"],
        y=df_contrib["Impact"],
        marker_color=colors
    ))
    fig.update_layout(
        title="Feature Impact (Red = Anomaly Driver, Green = Stabilizer)",
        yaxis_title="Impact",
    )
    st.plotly_chart(fig, use_container_width=True)

    csv_alert_df = display_alert_df.copy()
    csv = "IOT Sensor Health Report\n\n" + csv_alert_df.to_csv(index=False)

    st.download_button(
        label="Download Alerts CSV",
        data=csv,
        file_name="alerts.csv",
        mime="text/csv",
    )

    excel_bytes = None
    excel_error = None
    try:
        from openpyxl.styles import Font
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            csv_alert_df.to_excel(writer, index=False, sheet_name="Alerts")
            worksheet = writer.sheets["Alerts"]
            type_col_idx = csv_alert_df.columns.get_loc("System Status") + 1

            color_map = {
                "WARNING": "FFF59E0B",
                "HEALTHY": "FF16A34A",
                "NORMAL": "FF16A34A",
                "CRITICAL": "FFDC2626",
                "UNCERTAIN": "FF6B7280",
            }

            for row_idx in range(2, len(csv_alert_df) + 2):
                cell = worksheet.cell(row=row_idx, column=type_col_idx)
                status = str(cell.value).upper()
                if status in color_map:
                    cell.font = Font(color=color_map[status], bold=True)

        excel_bytes = excel_buffer.getvalue()
    except Exception as exc:
        excel_error = f"Colored Excel export unavailable: {exc}"

    if excel_bytes is not None:
        st.download_button(
            label="Download Alerts Excel (Colored)",
            data=excel_bytes,
            file_name="alerts_colored.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    elif excel_error is not None:
        st.caption(excel_error)

    if st.button("Generate PDF Report"):
        from reportlab.lib import colors as r_colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph

        def generate_pdf(dataframe):
            pdf_file = "alerts.pdf"
            doc = SimpleDocTemplate(pdf_file)
            styles = getSampleStyleSheet()
            title_style = styles["Title"]
            title_style.fontName = "Helvetica-Bold"
            title_style.alignment = 1

            normal_style = ParagraphStyle(
                "NormalCell",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=11,
            )

            severity_colors = {
                "NORMAL": r_colors.green,
                "WARNING": r_colors.orange,
                "CRITICAL": r_colors.red,
                "HEALTHY": r_colors.green,
            }

            total_cols = max(1, len(dataframe.columns))
            title_row = [Paragraph("IOT Sensor Health Report", title_style)] + [""] * (total_cols - 1)
            spacer_row = [""] * total_cols
            table_data = [title_row, spacer_row]

            header = [Paragraph(str(column), normal_style) for column in dataframe.columns]
            table_data.append(header)

            for _, row in dataframe.iterrows():
                row_cells = []
                for column in dataframe.columns:
                    value = row[column]
                    if column == "System Status":
                        color = severity_colors.get(str(value).upper(), r_colors.black)
                        status_style_p = ParagraphStyle(
                            "StatusCell",
                            parent=normal_style,
                            textColor=color,
                            fontName="Helvetica-Bold",
                        )
                        row_cells.append(Paragraph(str(value), status_style_p))
                    else:
                        row_cells.append(Paragraph(str(value), normal_style))
                table_data.append(row_cells)

            table = Table(
                table_data,
                repeatRows=3,
                colWidths=[95, 65, 55, 55, 60, 60]
            )
            table.setStyle(TableStyle([
                ("SPAN", (0, 0), (-1, 0)),
                ("GRID", (0, 2), (-1, -1), 0.75, r_colors.black),
                ("BOX", (0, 2), (-1, -1), 1.0, r_colors.black),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("BACKGROUND", (0, 2), (-1, 2), r_colors.HexColor("#D9E2F3")),
                ("ALIGN", (0, 2), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 2), (-1, 2), r_colors.HexColor("#BFD7EA")),
                ("BACKGROUND", (0, 3), (-1, -1), r_colors.white),
            ]))

            def _draw_footer(canvas, pdf_doc):
                footer_text = "Legend — Anomaly: 0..1 (higher = more anomalous). Health % = (1 - Anomaly) × 100 (higher = healthier)."
                canvas.saveState()
                canvas.setFont("Helvetica", 8)
                width, height = pdf_doc.pagesize
                canvas.drawCentredString(width / 2.0, 20, footer_text)
                canvas.restoreState()

            doc.build([table], onFirstPage=_draw_footer, onLaterPages=_draw_footer)
            return pdf_file
        
        file = generate_pdf(csv_alert_df)
        with open(file, "rb") as f:
            st.download_button(
                "Download PDF",
                f,
                file_name="alerts.pdf",
            )

if auto_refresh and not st.session_state["paused"]:
    time.sleep(refresh_seconds)
    st.rerun()
elif st.session_state["paused"]:
    time.sleep(0.5)
