# IoT AQI Manual (What Was Done and Why)

1. Converted the raw Excel sensor workbook into a clean tabular workflow for reproducible model training.
2. Verified data quality: no nulls, no duplicates, and consistent feature schema.
3. Sorted records by timestamp to preserve real temporal behavior before anomaly logic.
4. Built a rule-based detector from absolute rate-of-change and z-score thresholds.
5. Added hard-fault checks for saturation values (for example 4095) to capture sensor clipping quickly.
6. Split anomalies into spike anomalies and hardware failures for clearer root-cause interpretation.
7. Trained Isolation Forest to detect statistical outliers without requiring manual labels.
8. Tuned Isolation Forest hyperparameters using GridSearchCV to reduce over-detection.
9. Saved the best Isolation Forest model and score distribution for deployment reuse.
10. Standardized features with StandardScaler so all sensors contribute fairly to AE loss.
11. Trained an autoencoder on normal-like behavior to learn multivariate healthy patterns.
12. Used reconstruction MSE percentiles to flag deep pattern mismatches as AE anomalies.
13. Compared Rule, IF, and AE overlap to separate hard faults, statistical outliers, and subtle drifts.
14. Built a weighted fusion score combining rule, IF, and AE signals into one anomaly score.
15. Converted the fused score into operational alert bands: NORMAL, WARNING, CRITICAL.
16. Added local (hourly) AE thresholding to test temporal drift sensitivity.
17. Stored reusable artifacts: scaler.pkl, isolation_forest_model.pkl, autoencoder.keras, and .npy thresholds.
18. Implemented an explainable decision path so each alert can be traced to detector contributions.
19. Implemented the Streamlit app in app.py for real-time monitoring and model inference.
20. Added session state tracking for alerts, history, pause/stream control and latest sample context.
21. Added cached Excel loaders in app.py to avoid repeated heavy reads and improve responsiveness.
22. Added health and confidence views so operators can see both severity and agreement quality.
23. Added stability-aware logic to avoid overreacting to isolated transient jumps.
24. Net result: a hybrid, production-ready pipeline that detects clear faults, rare outliers and subtle behavior deviations with better reliability than any single detector.

### Notes

- Make sure the model and threshold files are present in the same folder as `app.py`.
- If you regenerate thresholds, run `thresh.py` before launching the dashboard.
- GridSearchCV: a scikit-learn tuning method that tries many hyperparameter combinations, evaluates each using cross-validation, and returns the best-performing model settings.
- StandardScaler: a preprocessing step that transforms each feature to zero mean and unit variance so large-range sensors do not dominate model learning.
- .pkl: a Python pickle file used to save and reload trained Python objects (for example the scaler and Isolation Forest model) without retraining.
- .npy: a NumPy binary array file used to store numeric data efficiently (for example score distributions and threshold values) with fast loading.
