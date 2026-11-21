# Project 2 – Machine-Learning-Based RF Signal Classification for Spectrum Sensing and Interference Identification

One click: open `RUN_ME.m` and press **Run**. Step-by-step instructions are in `HOW_TO_RUN.txt`.

## 1. What the project does (handout → code)

| Handout item | Where |
|---|---|
| Signals: 5G NR, LTE, Wi-Fi, Bluetooth, ZigBee (+ radar for "Radar vs 5G") | `p2_step1_waveform_pool.m`, `p2helpers/p2_gen*.m` |
| Progressive impairments: SNR → frequency offset → multipath → adjacent-channel → co-channel → multiple signals | `p2_step2_build_datasets.m`, `p2helpers/p2_makeFrame.m` (levels L1–L6) |
| Coexistence: BT vs Wi-Fi, ZigBee vs BT, ZigBee vs Wi-Fi, Radar vs 5G, LTE vs 5G | `p2helpers/p2_makeCoexFrame.m`, step 5d |
| Features: time, frequency, higher-order statistics, time–frequency (52) | `p2helpers/p2_extractFeatures.m` |
| Feature selection: correlation, mutual information, ANOVA F, RFE, PCA; minimum subset | `p2_step3_feature_selection.m` |
| Models: SVM, RF, KNN, DT, NB, LR, Gradient Boosting, AdaBoost, Hybrid RF+GB+SVM | `p2helpers/p2_trainModel.m`, `p2_step4_train_compare_models.m` |
| Metrics: accuracy, precision, recall, F1, confusion matrix, FPR, miss-detection, train/inference time, model size | `p2helpers/p2_metrics.m`, step 4 |
| Performance vs SNR, bandwidth, frequency separation, interference level | `p2_step5_robustness.m` |
| SDR validation and sim-to-real gap | `p2_step6_sdr_validation.m` |
| Online learning, band transfer, low-SNR, real-time inference, model comparison | `p2_step7_advanced.m`, step 5e |

## 2. Signals (fs = 30.72 MHz, frame = 8192 samples = 0.267 ms)
Noise (no signal), WLAN 802.11a/g/n 20 MHz, Bluetooth BR/EDR + LE, ZigBee 802.15.4 O-QPSK, LTE 1.4–20 MHz, 5G NR 5–20 MHz, pulsed radar (LFM chirps, Barker-13). Toolbox generators are used when installed, and the project's own standard-based generators are used otherwise. The generator used is listed in `results/p2_step1_pool_summary.csv`.

## 3. Gradient boosting / XGBoost
MATLAB has no XGBoost. The model `GradientBoosting` uses LogitBoost trees (gradient boosting with the logistic loss) in a one-vs-all ECOC scheme. `p2_xgboost_optional.py` trains real XGBoost on exactly the same features and split.

## 4. Output
`results/` holds every figure (`.png`), table (`.csv`), `RESULTS_SUMMARY.txt` and `command_window_log.txt`. `Project2_results_to_send.zip` contains all of these files.

## 5. Settings
All settings live in `p2Config.m`. `MODE = "quick"` in `RUN_ME.m` makes everything small, for a quick check that the code runs.

## 6. SDR validation (ADALM-PLUTO)
Install the ADALM-Pluto support package. In `p2Config.m` set `cfg.sdr.mode` to one of:
* `"labeled"` – TX → **30 dB attenuator** → RX cable. Each class is transmitted and received, and accuracy on real hardware is compared with the simulated accuracy (the sim-to-real gap). Never transmit over the air without a licence.
* `"ambient"` – antenna on RX only, at 2.44 GHz by default. Reports what is on air.
* `"recordings"` – put MAT-files with `x` (complex samples, 30.72 MHz) and optionally `label` in `captures/`. USRP or any other SDR recordings work this way.

Then run `RUN_FINISH.m` with `START_OVERRIDE = 6`. Without a radio or recordings, step 6 is skipped and nothing is simulated in its place.
