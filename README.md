# RF Signal Classification for Spectrum Sensing

M.Tech project work — Electronics and Communication Engineering (Communication Systems), MANIT Bhopal.

| Folder | What it is | How to run |
|---|---|---|
| `project1-deep-learning-matlab/` | **Project 1** – Bluetooth + WiFi spectrogram segmentation by transfer learning from the MathWorks 5G/LTE spectrum-sensing network | MATLAB: open `RUN_ME.m`, press Run |
| `project2-machine-learning-matlab/` | **Project 2** – 52 handcrafted features and 9 shallow classifiers (SVM, RF, KNN, DT, NB, LR, Gradient Boosting, AdaBoost, Hybrid) for 5G NR, LTE, WiFi, Bluetooth, ZigBee and radar under 6 impairment levels and 5 coexistence scenarios | MATLAB: open `RUN_ME.m`, press Run (`MODE = "quick"` for a short test) |
| `presentation-app/` | Streamlit app: results of both projects and a live demo for each | `START_APP.command` (macOS) / `START_APP.bat` (Windows), or `pip install -r requirements.txt` then `streamlit run app.py` |

## Key results (MATLAB runs)
- **Project 1:** 98.2 % pixel accuracy, mean IoU 0.80 (Bluetooth 0.56, WLAN 0.87), 99.5 % recall for both signals.
- **Project 2:** Random Forest best — 84 % (level 1) → 62.7 % (level 6), > 90 % above ≈14 dB SNR; recursive feature elimination keeps full accuracy with 10 of 52 features.

## Requirements
- MATLAB R2022a+ with Communications, Signal Processing, Bluetooth, Deep Learning (Project 1) and Statistics & Machine Learning (Project 2) Toolboxes. WLAN, LTE, 5G and Parallel Computing Toolboxes are optional.
- Python 3.9+ for the presentation app.

Generated data, trained models and results are not stored in the repository; `RUN_ME.m` recreates them.
