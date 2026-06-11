# load your saved IF distribution (already inverted: higher = more anomalous)
import os
import numpy as np
if_dist = np.load("if_score_dist.npy")

thresholds = {
    "normal_max": np.percentile(if_dist, 70),
    "slightly_unusual": np.percentile(if_dist, 85),
    "suspicious": np.percentile(if_dist, 95),
    "critical": np.percentile(if_dist, 99),
}

np.save("if_thresholds.npy", thresholds)

ae_dist = np.load("ae_error_dist.npy")

ae_thresholds = {
    "slight": np.percentile(ae_dist, 85),
    "suspicious": np.percentile(ae_dist, 95),
    "critical": np.percentile(ae_dist, 99),
}

np.save("ae_thresholds.npy", ae_thresholds)

score_dist = np.load("score_dist.npy")

score_thresholds = {
    "normal": np.percentile(score_dist, 40),
    "warning": np.percentile(score_dist, 88),
    "critical": np.percentile(score_dist, 99),
}

np.save("score_thresholds.npy", score_thresholds)


def _anomaly_to_health_percent(anomaly_score):
    # Anomaly score is in [0, 1] where higher means worse; health in [0, 100] where higher means better.
    return float(np.clip((1.0 - float(anomaly_score)) * 100.0, 0.0, 100.0))


health_thresholds = {
    "critical": _anomaly_to_health_percent(score_thresholds["critical"]),
    "warning": _anomaly_to_health_percent(score_thresholds["warning"]),
    "healthy": _anomaly_to_health_percent(score_thresholds["normal"]),
}

# Keep both names for backward compatibility across app versions.
np.save("health_threshold.npy", health_thresholds)
np.save("health_thresholds.npy", health_thresholds)