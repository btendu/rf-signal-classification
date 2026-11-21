"""RF Signal Classification - presentation app for Project 1 and Project 2.

Start:  streamlit run app.py      (or double-click START_APP)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import p1demo as P1
import rfdemo as R

HERE = os.path.dirname(os.path.abspath(__file__))
A1 = os.path.join(HERE, "assets", "p1")
A2 = os.path.join(HERE, "assets", "p2")
MODEL_FILE = os.path.join(HERE, "demo_model.joblib")
P1_MODEL_FILE = os.path.join(HERE, "demo_model_p1.joblib")
NAVY = "#1F3864"
CLASS_COLORS = {"Noise": "#9AA5B1", "WLAN": "#1F77B4", "Bluetooth": "#2CA02C", "ZigBee": "#9467BD",
                "LTE": "#FF7F0E", "NR5G": "#D62728", "Radar": "#8C564B"}

st.set_page_config(page_title="RF Signal Classification", page_icon="📡", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1300px;}
  h1, h2, h3 {color: %s;}
  .big {font-size: 1.15rem; line-height: 1.6;}
  .card {border: 1px solid #D0D7E2; border-radius: 10px; padding: 1rem 1.2rem; background: #F7F9FC; height: 100%%;}
  .card h4 {margin-top: 0; color: %s;}
  .ok {color: #1B7F3B; font-weight: 700;}
  .bad {color: #B42318; font-weight: 700;}
  div[data-testid="stMetricValue"] {color: %s;}
  .kpis {display: flex; gap: .7rem; flex-wrap: wrap; margin: .3rem 0 1rem 0;}
  .kpi {flex: 1 1 150px; border: 1px solid #D0D7E2; border-radius: 10px; padding: .6rem .9rem; background: #F7F9FC;}
  .kpi .l {font-size: .8rem; color: #555;}
  .kpi .v {font-size: 1.35rem; font-weight: 700; color: %s; line-height: 1.3;}
</style>
""" % (NAVY, NAVY, NAVY, NAVY), unsafe_allow_html=True)


def kpis(items):
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v in items) + "</div>",
        unsafe_allow_html=True)


def img_narrow(folder, name, caption=None, frac=0.75):
    pad = (1 - frac) / 2
    _, mid, _ = st.columns([pad, frac, pad])
    with mid:
        img(folder, name, caption)


def img(folder, name, caption=None):
    p = os.path.join(folder, name)
    if os.path.isfile(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info(f"Figure {name} not found in {folder}.")


# ======================================================================= data
P2_LEVEL = {  # accuracy (%) L1-L5 from the MATLAB figures, L6 exact
    "SVM": [74.2, 64.1, 59.3, 51.7, 46.9], "Random Forest": [84.2, 77.1, 74.0, 64.1, 66.7],
    "KNN": [71.3, 56.5, 50.2, 41.4, 44.6], "Decision Tree": [76.2, 73.1, 64.9, 51.4, 51.4],
    "Naive Bayes": [57.6, 54.3, 52.6, 43.4, 40.3], "Logistic Regression": [79.6, 71.0, 70.6, 55.8, 56.4],
    "Gradient Boosting": [85.0, 79.8, 74.9, 63.4, 65.7], "AdaBoost": [80.6, 74.4, 72.8, 57.3, 57.4],
    "Hybrid (RF+GB+SVM)": [84.0, 78.9, 74.9, 64.0, 65.8]}
KEYMAP = {"SVM": "SVM", "Random Forest": "RandomForest", "KNN": "KNN", "Decision Tree": "DecisionTree",
          "Naive Bayes": "NaiveBayes", "Logistic Regression": "LogisticRegression",
          "Gradient Boosting": "GradientBoosting", "AdaBoost": "AdaBoost", "Hybrid (RF+GB+SVM)": "Hybrid"}
P2_COEX = {  # BT-WLAN, ZB-BT, ZB-WLAN, Radar-NR, LTE-NR
    "SVM": [76.6, 75.6, 78.2, 76.0, 58.3], "Random Forest": [84.0, 89.3, 83.5, 84.3, 72.7],
    "KNN": [75.0, 67.3, 74.5, 71.2, 56.3], "Decision Tree": [77.3, 81.3, 82.2, 81.3, 65.0],
    "Naive Bayes": [78.3, 68.6, 79.6, 75.6, 57.7], "Logistic Regression": [84.0, 84.6, 87.7, 82.3, 67.9],
    "Gradient Boosting": [83.2, 87.6, 84.9, 87.0, 70.7], "AdaBoost": [81.3, 88.6, 84.2, 86.6, 68.4],
    "Hybrid (RF+GB+SVM)": [85.0, 87.3, 86.2, 86.0, 72.7]}
P2_TIME = {"SVM": (1.4, 380), "Random Forest": (2.0, 97), "KNN": (0.035, 20), "Decision Tree": (0.05, 2.4),
           "Naive Bayes": (0.012, 1.9), "Logistic Regression": (0.30, 14), "Gradient Boosting": (3.9, 2250),
           "AdaBoost": (1.35, 39), "Hybrid (RF+GB+SVM)": (7.3, 2750)}
try:
    with open(os.path.join(A2, "metrics.json")) as fh:
        P2_M = json.load(fh)
except Exception:
    P2_M = {}


# ======================================================================= pages
def page_overview():
    st.title("📡 RF Signal Classification for Spectrum Sensing")
    st.markdown('<p class="big">M.Tech project work, Electronics and Communication Engineering '
                '(Communication Systems), MANIT Bhopal. Two projects, one question: '
                '<b>which wireless systems are on the air, and who is interfering with whom?</b></p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="card"><h4>Project 1 · Deep learning</h4>
        <b>Bluetooth and WiFi spectrogram segmentation by transfer learning</b><br><br>
        • A pretrained 5G/LTE segmentation network is retrained to label every pixel of a 20 ms,
          60 MHz spectrogram as Noise, Bluetooth or WLAN.<br>
        • 1500 synthetic frames with fading, CFO, phase noise and DC offset.<br>
        • <b>98.2 % pixel accuracy, mean IoU 0.80</b>; 99.5 % of Bluetooth and WiFi pixels found.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><h4>Project 2 · Machine learning</h4>
        <b>Handcrafted features + shallow classifiers for 5G, LTE, WiFi, Bluetooth, ZigBee and radar</b><br><br>
        • 52 time, frequency, higher-order and time–frequency features per 0.267 ms frame.<br>
        • 9 models, 6 impairment levels, 5 coexistence scenarios, feature selection.<br>
        • <b>Random Forest best: 84 % (L1) → 62.7 % (L6), &gt;90 % above 14 dB SNR</b>; 10 of 52 features suffice.
        </div>""", unsafe_allow_html=True)
    st.write("")
    st.subheader("Project 2 processing chain")
    img_narrow(A2, "d_pipeline.png", frac=0.7)
    st.subheader("How to use this app")
    st.markdown("""
- **Project 1** and **Project 2** pages: the results of the MATLAB runs, step by step.
- **Live demo · Project 1**: build a Bluetooth + WiFi frame and watch every spectrogram pixel being
  labelled Noise, Bluetooth or WLAN, with accuracy and IoU.
- **Live demo · Project 2**: pick a signal, SNR, frequency offset and interferers; the app builds the signal,
  extracts the features and a Random Forest classifies it on the spot.
- Use the sidebar to switch pages. Press **R** (or the ⋮ menu → Rerun) to refresh.
""")


def page_p1():
    st.title("Project 1 · Deep-Learning Spectrogram Segmentation")
    st.caption("Transfer learning from the MathWorks 5G/LTE spectrum-sensing network to Bluetooth + WiFi (MATLAB, CPU).")
    kpis([("Global accuracy", "98.23 %"), ("Mean accuracy", "99.06 %"), ("Mean IoU", "0.804"),
          ("WLAN IoU", "0.867"), ("Bluetooth IoU", "0.563")])
    tabs = st.tabs(["1 · Signals", "2 · Combined band", "3 · Training data", "4 · Training",
                    "5 · Test results", "6 · Overlap study", "Summary"])
    with tabs[0]:
        st.markdown("Bluetooth BR/EDR (Bluetooth Toolbox) and 802.11 WiFi (own generator) waveforms.")
        img(A1, "r_step1.png", "Amplitude and spectrum of the Bluetooth (top) and WiFi (bottom) waveforms.")
    with tabs[1]:
        st.markdown("Both signals are moved to their own frequencies with the **Multiband Combiner**: "
                    "Bluetooth at −15 MHz, WiFi at +12 MHz, 17.6 MHz apart (no overlap).")
        c1, c2 = st.columns(2)
        with c1:
            img(A1, "r_step2_psd.png", "Spectrum before and after combining.")
        with c2:
            img(A1, "r_step2_spec.png", "Spectrogram of the combined signal at 2.44 GHz.")
    with tabs[2]:
        st.markdown("1500 training frames (20 ms, 60 MHz): Bluetooth only 25 %, WiFi only 25 %, both 45 %, "
                    "noise 5 %, in-band SNR 5–40 dB. Each frame comes with a pixel label map.")
        img(A1, "r_train_both.png", "Training frame with both signals and its pixel labels.")
        c1, c2 = st.columns(2)
        with c1:
            img(A1, "r_train_bt.png", "Bluetooth only.")
        with c2:
            img(A1, "r_train_wlan.png", "WiFi only.")
        img_narrow(A1, "r_balance.png", "Bluetooth covers only 0.71 % of pixels, so class weighting is needed.", 0.5)
    with tabs[3]:
        c1, c2 = st.columns([3, 2])
        with c1:
            img(A1, "r_training.png", "Training and validation loss.")
        with c2:
            st.markdown("""
| Item | Value |
|---|---|
| Layers reused from the 5G/LTE network | 114 of 118 |
| Epochs / iterations | 12 / 900 (early stop) |
| Final training loss | 0.0273 |
| Best validation loss | 0.059 (epoch 7) |
| Training time | 36.7 min on CPU |
""")
    with tabs[4]:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("""
| Class | Recall | IoU |
|---|---|---|
| Noise | 0.981 | 0.981 |
| Bluetooth | 0.995 | 0.563 |
| WLAN | 0.995 | 0.867 |
""")
            st.markdown("No Bluetooth pixel is mistaken for WiFi or vice versa. The lower Bluetooth IoU comes "
                        "from labels at packet edges and from IQ-imbalance mirror images marked as Bluetooth.")
        with c2:
            img(A1, "r_confusion.png", "Pixel confusion matrix (150 test frames).")
        sel = st.radio("Test frame", ["Frame 1 · WiFi only", "Frame 4 · weak Bluetooth", "Frame 7 · both",
                                      "Frame 150 · both, strong"], horizontal=True)
        img(A1, {"Frame 1 · WiFi only": "r_test1.png", "Frame 4 · weak Bluetooth": "r_test4.png",
                 "Frame 7 · both": "r_test7.png", "Frame 150 · both, strong": "r_test150.png"}[sel],
            "Spectrogram, true labels and predicted labels.")
        img(A1, "r_iou_snr.png", "Mean IoU per frame against the SNR of the weakest signal.")
    with tabs[5]:
        st.markdown("Network trained on **non-overlapping** frames, tested on 150 frames where Bluetooth may hop "
                    "**onto the WiFi channel**.")
        st.table(pd.DataFrame({"Metric": ["Global accuracy", "Mean IoU", "Bluetooth recall", "Bluetooth IoU",
                                          "WLAN recall", "WLAN IoU"],
                               "No overlap": [0.9823, 0.8035, 0.9954, 0.5626, 0.9954, 0.8672],
                               "With overlap": [0.9734, 0.7760, 0.7824, 0.5075, 0.9917, 0.8486]}).set_index("Metric"))
        img(A1, "r_overlap.png", "Overlap test frame: spectrogram, truth and prediction.")
    with tabs[6]:
        st.markdown("""
- Transfer learning reused 114 of 118 parameter arrays; only the output layers were trained from scratch.
- **98.2 %** of pixels correct, **mean IoU 0.80**, 99.5 % recall for both Bluetooth and WiFi.
- WiFi stays robust when Bluetooth overlaps it (recall 99.2 %); Bluetooth recall drops to 78 %.
- Over-the-air validation with an ADALM-PLUTO is prepared in code; no radio was available for this run.
""")


def page_p2():
    st.title("Project 2 · Machine-Learning RF Signal Classification")
    st.caption("52 handcrafted features · 9 classifiers · 6 impairment levels · 5 coexistence scenarios (MATLAB full run).")
    kpis([("Best model", "Random Forest"), ("Accuracy L1 → L6", "84 % → 62.7 %"),
          ("90 % accuracy above", "≈14 dB SNR"), ("Minimum feature subset", "10 of 52 (RFE)"),
          ("False alarms at L6", "2 %")])
    tabs = st.tabs(["Signals & levels", "Feature selection", "Model comparison", "Level-6 details",
                    "SNR & bandwidth", "Interference", "Coexistence", "Advanced", "Conclusions"])
    with tabs[0]:
        img(A2, "step1_waveforms.png", "One generated waveform of each class: spectrogram (top) and spectrum (bottom).")
        img_narrow(A2, "d_levels.png", "The six progressive impairment levels.", 0.8)
        lv = st.radio("Example frames", ["Level 1", "Level 6"], horizontal=True)
        img(A2, "step2_examples_L1.png" if lv == "Level 1" else "step2_examples_L6.png")
    with tabs[1]:
        c1, c2 = st.columns([3, 2])
        with c1:
            img(A2, "step3_acc_vs_features.png", "Cross-validated accuracy against number of features.")
        with c2:
            st.markdown("""
**Minimum subset: 10 features (RFE)**, same accuracy as all 52.

- **RFE** is best at small k (ranks by usefulness to the model).
- **Correlation** filter and **ANOVA** follow closely.
- **MRMR** mutual information is weak for k < 20.
- **PCA** is worst (≈53 %): it keeps variance, not class separation; 22 components needed for 95 % variance.
""")
        img(A2, "step3_rankings.png", "ANOVA F-score, MRMR score and PCA explained variance.")
        img(A2, "step3_correlation.png", "Feature correlation: 39 of 52 features kept at |r| < 0.9.")
    with tabs[2]:
        df = pd.DataFrame({k: v + [100 * P2_M.get(KEYMAP[k], {}).get("acc", np.nan)] for k, v in P2_LEVEL.items()},
                          index=["L1", "L2", "L3", "L4", "L5", "L6"]).T
        sel = st.multiselect("Models to show", list(df.index),
                             default=["Random Forest", "Gradient Boosting", "Hybrid (RF+GB+SVM)", "Logistic Regression", "SVM", "Naive Bayes"])
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for mname in sel:
            ax.plot(df.columns, df.loc[mname], "-o", lw=2, label=mname)
        ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(30, 90); ax.grid(alpha=.35)
        ax.legend(fontsize=8, ncol=2); ax.set_title("Accuracy as impairments are added")
        st.pyplot(fig, use_container_width=True)
        st.dataframe(df.style.format("{:.1f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)
        st.caption("L1–L5 measured from the MATLAB figure (±0.5 %); L6 computed exactly from the confusion matrices.")
        c1, c2 = st.columns(2)
        with c1:
            img(A2, "step4_timing.png", "Training and inference time.")
        with c2:
            img(A2, "step5e_acc_latency.png", "Accuracy vs latency (marker size = model size).")
    with tabs[3]:
        if P2_M:
            rows = []
            for k, key in KEYMAP.items():
                r = P2_M[key]
                rows.append([k, 100 * r["acc"], 100 * r["prec"], 100 * r["rec"], 100 * r["f1"], 100 * r["fpr"],
                             100 * r["miss"], 100 * r["falseAlarm"], 100 * r["missedAsNoise"]])
            t = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "FPR", "Miss rate",
                                            "False alarm", "Missed as noise"]).set_index("Model")
            st.dataframe(t.style.format("{:.1f}").highlight_max(subset=["Accuracy", "Precision", "Recall", "F1"],
                                                                 color="#D6E6F7"), use_container_width=True)
            st.caption("All in %, macro averages over 7 classes, 700 test frames — exact values from the confusion matrices.")
        c1, c2 = st.columns(2)
        with c1:
            img(A2, "step4_confusion_best.png", "Random Forest, level 6.")
        with c2:
            img(A2, "d_f1_heatmap.png", "F1 per class and model.")
        st.markdown("**Key finding:** LTE and 5G NR are the hardest pair (35 % of LTE called NR) — both are CP-OFDM "
                    "with flat spectra and identical statistics; in this run they also came from the same OFDM generator.")
        with st.expander("All nine confusion matrices"):
            img(A2, "step4_confusion_all.png")
    with tabs[4]:
        img(A2, "step5a_snr.png", "Accuracy vs in-band SNR (left) and per-class recall of Random Forest (right).")
        st.markdown("- Near chance below −6 dB · linear rise to 10 dB · **RF, GB, Hybrid reach 90 % at ≈14 dB**.\n"
                    "- Radar easiest (82 % at 0 dB); LTE never above ≈80 % (confused with NR).")
        img(A2, "step5b_bw.png", "Accuracy vs bandwidth: LTE/NR confusion happens where their bandwidths are equal.")
    with tabs[5]:
        img(A2, "step5c_interf.png", "Accuracy vs SIR of the strongest interferer and vs number of interferers.")
        st.markdown("Random Forest: ≈47 % at SIR 0–5 dB → 64–68 % above 10 dB; 60 % → 54 % from 1 to 3 interferers.")
    with tabs[6]:
        names = ["BT vs WLAN", "ZigBee vs BT", "ZigBee vs WLAN", "Radar vs 5G", "LTE vs 5G"]
        cdf = pd.DataFrame(P2_COEX, index=names).T
        st.dataframe(cdf.style.format("{:.1f}").highlight_max(axis=0, color="#D6E6F7"), use_container_width=True)
        st.caption("Accuracy (%) with 4 labels: A, B, A+B, noise.")
        k = st.selectbox("Confusion matrix of the best model", names)
        c1, c2 = st.columns([2, 3])
        with c1:
            img(A2, f"coex{names.index(k) + 1}.png")
        with c2:
            img(A2, "step5d_sep_sir.png", "Both-signal detection vs frequency separation and power ratio.")
        st.markdown("Well-separated signals are both found in up to 100 % of frames; strong overlap and a 10 dB "
                    "power difference hide the weaker one (masking).")
    with tabs[7]:
        sub = st.radio("Extension", ["Online learning", "Band transfer", "Low SNR", "Real time"], horizontal=True)
        f = {"Online learning": ("step7a_incremental.png", "Model trained on L1 adapts to L6 within ≈100 frames (29 % → 42 %), "
                                 "reaching the Naive Bayes ceiling."),
             "Band transfer": ("step7b_transfer.png", "Random Forest loses nothing when moved to the other half-band; SVM loses 7–10 points."),
             "Low SNR": ("step7c_lowsnr.png", "Low-SNR specialist gains slightly below 0 dB but loses badly above 4 dB."),
             "Real time": ("step7d_realtime.png", "Feature extraction ≈2.3 ms per frame > 0.267 ms frame: not real-time in MATLAB on one core.")}[sub]
        img(A2, f[0]); st.markdown(f[1])
    with tabs[8]:
        st.markdown("""
1. Handcrafted features + shallow ML recognise 6 signal types: **> 90 % above ≈14 dB SNR**.
2. Every impairment costs accuracy; frequency offset and adjacent-channel interference cost most.
3. **Tree ensembles win; Random Forest** gives the best accuracy–cost balance (≈0.1 ms/frame).
4. **RFE: 10 of 52 features** keep full accuracy; PCA is unsuitable.
5. Coexistence: 85–89 % for four pairs, 73 % for LTE vs 5G.
6. LTE vs 5G NR is the hardest pair — needs standard-specific features (CP, sync signals).
7. Online learning adapts fast; tree models transfer across bands; features are the real-time bottleneck.
""")


@st.cache_resource(show_spinner=False)
def get_model():
    bar = st.progress(0.0, text="First start: generating 1540 training frames and training the Random Forest…")
    d = R.load_or_train(MODEL_FILE, progress=lambda p: bar.progress(min(p, 1.0), text=f"Building training data… {int(p * 100)} %"))
    bar.empty()
    return d


def plot_frame(x, info, fs=R.FS):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    from scipy import signal as sps
    f, t, S = sps.spectrogram(x, fs=fs, nperseg=128, noverlap=96, return_onesided=False, mode="psd")
    S = np.fft.fftshift(S, axes=0); f = np.fft.fftshift(f)
    SdB = 10 * np.log10(S + 1e-20)
    fl = np.percentile(SdB, 10)
    axs[0].pcolormesh(t * 1e3, f / 1e6, SdB, shading="auto", cmap="turbo", vmin=fl - 3, vmax=fl + 40)
    axs[0].set_xlabel("Time (ms)"); axs[0].set_ylabel("Frequency (MHz)"); axs[0].set_title("Spectrogram")
    fw, P = sps.welch(x, fs=fs, nperseg=512, return_onesided=False)
    P = np.fft.fftshift(P); fw = np.fft.fftshift(fw)
    axs[1].plot(fw / 1e6, 10 * np.log10(P + 1e-20), color=NAVY, lw=1)
    if info.get("bw"):
        axs[1].axvspan((info["fc"] - info["bw"] / 2) / 1e6, (info["fc"] + info["bw"] / 2) / 1e6,
                       color=CLASS_COLORS.get(info["class"], "#ccc"), alpha=.18, label=info["class"])
    for it in info.get("interferers", []):
        axs[1].axvline(it["fc"] / 1e6, color=CLASS_COLORS[it["class"]], ls="--", lw=1)
    axs[1].set_xlabel("Frequency (MHz)"); axs[1].set_ylabel("PSD (dB/Hz)"); axs[1].set_title("Power spectrum")
    axs[1].grid(alpha=.3)
    if info.get("bw"):
        axs[1].legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def get_model_p1():
    bar = st.progress(0.0, text="First start: generating 90 Bluetooth/WiFi training frames and training the pixel classifier…")
    d = P1.load_or_train(P1_MODEL_FILE, progress=lambda p: bar.progress(min(p, 1.0), text=f"Building training data… {int(p * 100)} %"))
    bar.empty()
    return d


def plot_p1(f, t, S, Lt, Lp):
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#E9EDF2", CLASS_COLORS["Bluetooth"], CLASS_COLORS["WLAN"]])
    ext = [t[0] * 1e3, t[-1] * 1e3, f[0] / 1e6, f[-1] / 1e6]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    SdB = 10 * np.log10(S + 1e-20)
    fl = np.percentile(SdB, 20)
    axs[0].imshow(SdB, aspect="auto", origin="lower", extent=ext, cmap="turbo", vmin=fl - 3, vmax=fl + 35)
    axs[0].set_title("Received spectrogram")
    axs[1].imshow(Lt, aspect="auto", origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axs[1].set_title("True labels")
    axs[2].imshow(Lp, aspect="auto", origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axs[2].set_title("Predicted labels")
    axs[0].set_ylabel("Frequency (MHz)")
    for a in axs:
        a.set_xlabel("Time (ms)")
    from matplotlib.patches import Patch
    axs[2].legend(handles=[Patch(color="#E9EDF2", label="Noise"), Patch(color=CLASS_COLORS["Bluetooth"], label="Bluetooth"),
                           Patch(color=CLASS_COLORS["WLAN"], label="WLAN")], loc="upper right", fontsize=8, framealpha=.9)
    fig.tight_layout()
    return fig


def page_demo1():
    st.title("🔴 Live demo · Project 1 · pixel labelling")
    st.caption("Same task as Project 1: label every pixel of a wideband spectrogram (60 MHz around 2.44 GHz) as Noise, "
               "Bluetooth or WLAN. The MATLAB deep network cannot run in Python, so this demo uses a lighter pixel "
               "classifier (gradient-boosted trees on 12 local time–frequency features).")
    d = get_model_p1()
    mdl = d["model"]
    with st.sidebar:
        st.markdown("### Frame settings")
        content = st.radio("Signals present", ["Both", "Bluetooth", "WLAN", "Noise"], horizontal=True)
        snr_bt = st.slider("Bluetooth in-band SNR (dB)", 0, 40, 20, disabled=content in ("WLAN", "Noise"))
        snr_w = st.slider("WiFi in-band SNR (dB)", 0, 40, 20, disabled=content in ("Bluetooth", "Noise"))
        wfc = st.slider("WiFi channel centre (MHz)", -18.0, 18.0, 12.0, 1.0, disabled=content in ("Bluetooth", "Noise"))
        ov = st.checkbox("Let Bluetooth hop onto the WiFi channel (overlap)", value=False,
                         disabled=content != "Both")
        seed = st.number_input("Random seed", 0, 10_000, 4, key="seed1", help="Change it to get a new random frame")
    tab1, tab2, tab3 = st.tabs(["Single frame", "Batch test", "About the model"])
    with tab1:
        rng = np.random.default_rng(int(seed))
        x, tx = P1.make_frame(rng, content, snr_bt=snr_bt, snr_wlan=snr_w, overlap=ov, wlan_fc=wfc * 1e6)
        t0 = time.perf_counter()
        f, t, S = P1.spectrogram(x)
        Lp = P1.predict_map(mdl, S)
        dt = time.perf_counter() - t0
        Lt = P1.true_labels(tx, f, t)
        acc, iou = P1.metrics(Lt, Lp)
        fmt = lambda v: "—" if np.isnan(v) else f"{v:.3f}"
        kpis([("Pixel accuracy", f"{100 * acc:.2f} %"), ("Noise IoU", fmt(iou["Noise"])),
              ("Bluetooth IoU", fmt(iou["Bluetooth"])), ("WLAN IoU", fmt(iou["WLAN"])),
              ("Processing time", f"{dt:.1f} s")])
        st.pyplot(plot_p1(f, t, S, Lt, Lp), use_container_width=True)
        nbt = sum(1 for c in tx if c[0] == "Bluetooth")
        nw = sum(1 for c in tx if c[0] == "WLAN")
        st.caption(f"This frame: {nbt} Bluetooth packets (366 µs, hopping on 1 MHz channels) and {nw} WiFi packets "
                   f"(16.6 MHz) in {P1.T * 1e3:.0f} ms. IoU = overlap of predicted and true region of a class.")
    with tab2:
        st.markdown("Generate random frames (SNR 5–40 dB, random positions) and measure the pixel accuracy and IoU, "
                    "as in Project 1's test set.")
        c1, c2 = st.columns(2)
        nfr = c1.slider("Number of frames", 3, 30, 9)
        bov = c2.checkbox("Allow overlap", value=False, key="bov")
        if st.button("▶ Run batch test", type="primary", key="b1"):
            rng = np.random.default_rng(int(seed) + 500)
            accs, ious = [], {k: [] for k in P1.LABELS}
            C = np.zeros((3, 3))
            bar = st.progress(0.0)
            for i in range(nfr):
                xx, txx = P1.make_frame(rng, ["Both", "Bluetooth", "WLAN"][i % 3], overlap=bov)
                ff, tt, SS = P1.spectrogram(xx)
                lt, lp = P1.true_labels(txx, ff, tt), P1.predict_map(mdl, SS)
                a, io = P1.metrics(lt, lp)
                accs.append(a)
                for k in P1.LABELS:
                    if not np.isnan(io[k]):
                        ious[k].append(io[k])
                for i1 in range(3):
                    for j1 in range(3):
                        C[i1, j1] += np.sum((lt == i1) & (lp == j1))
                bar.progress((i + 1) / nfr)
            bar.empty()
            kpis([("Global accuracy", f"{100 * np.mean(accs):.2f} %"),
                  ("Mean IoU", f"{np.mean([np.mean(v) for v in ious.values() if v]):.3f}"),
                  ("Bluetooth IoU", f"{np.mean(ious['Bluetooth']):.3f}"), ("WLAN IoU", f"{np.mean(ious['WLAN']):.3f}")])
            Cn = pd.DataFrame(100 * C / C.sum(1, keepdims=True), index=["True " + k for k in P1.LABELS],
                              columns=["Pred. " + k for k in P1.LABELS])
            st.dataframe(Cn.style.format("{:.2f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)
            st.caption("Pixel confusion matrix (%, row-normalised). Compare: MATLAB network 98.2 % accuracy, "
                       "Bluetooth IoU 0.563, WLAN IoU 0.867.")
    with tab3:
        st.markdown(f"""
- **Task:** semantic segmentation of a {P1.T * 1e3:.0f} ms, 61.44 MHz spectrogram ({P1.NFFT}-point FFT, 240 kHz × 4.2 µs pixels).
- **Signals:** Bluetooth GFSK packets (1 Msym/s, h = 0.32) in 625 µs slots hopping over 1 MHz channels;
  802.11 OFDM WiFi packets on one 20 MHz channel; in-band SNR 5–40 dB.
- **Model:** histogram gradient boosting, 200 trees, on 12 features per pixel (power, local means over
  3–65 bins in frequency and 9–41 slices in time, local max/std, channel persistence).
- **Training:** {d['n_pixels']:,} balanced pixels from 90 frames, trained in {d['train_time']:.0f} s.
- **Held-out test (15 frames):** accuracy {100 * d['test_acc']:.2f} %, IoU Noise {d['test_iou']['Noise']:.3f},
  Bluetooth {d['test_iou']['Bluetooth']:.3f}, WLAN {d['test_iou']['WLAN']:.3f}.

The MATLAB deep network of Project 1 reached 98.2 % / IoU 0.563 (Bluetooth) / 0.867 (WLAN); the reported
numbers are those of the MATLAB run. Bluetooth IoU is low in both because Bluetooth regions are tiny
(< 1 % of pixels) and every edge pixel counts.
""")
        if st.button("Retrain the Project 1 demo model"):
            if os.path.isfile(P1_MODEL_FILE):
                os.remove(P1_MODEL_FILE)
            get_model_p1.clear()
            st.rerun()


def page_demo():
    st.title("🔴 Live demo · Project 2 · classify a signal")
    st.caption("Python re-implementation of the Project 2 method: simplified standard-numerology waveforms, the same "
               "frame (8192 samples at 30.72 MHz), 41 handcrafted features and a Random Forest trained on all six levels.")
    d = get_model()
    mdl = d["model"]
    with st.sidebar:
        st.markdown("### Signal settings")
        cls = st.selectbox("Transmitted signal", R.CLASSES, index=1)
        snr = st.slider("In-band SNR (dB)", -10, 30, 15)
        off = st.slider("Frequency offset (MHz)", -14.0, 14.0, 0.0, 0.5)
        mp = st.checkbox("Multipath fading", value=False)
        st.markdown("### Interference")
        n_int = st.radio("Interferers", [0, 1, 2, 3], horizontal=True)
        ints = []
        for i in range(n_int):
            c1, c2 = st.columns(2)
            ic = c1.selectbox(f"Interferer {i + 1}", [c for c in R.SIGNALS if c != cls], key=f"ic{i}", index=i % 5)
            ik = c2.selectbox("Type", ["adjacent", "co-channel"], key=f"ik{i}")
            ints.append((ic, ik))
        sir = st.slider("Signal-to-interference ratio (dB)", 0, 20, 10, disabled=n_int == 0)
        seed = st.number_input("Random seed", 0, 10_000, 7, help="Change it to get a new random frame")
    tab_one, tab_batch, tab_model = st.tabs(["Single frame", "Batch test", "About the model"])
    with tab_one:
        rng = np.random.default_rng(int(seed))
        x, info = R.make_frame(rng, cls, snr_db=snr, level=1, offset_hz=off * 1e6, interferers=ints,
                               sir_db=sir, multipath=mp)
        t0 = time.perf_counter()
        v, feats = R.features_vector(x)
        t1 = time.perf_counter()
        proba = mdl.predict_proba([v])[0]
        t2 = time.perf_counter()
        labels = list(mdl.classes_)
        pred = labels[int(np.argmax(proba))]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transmitted", cls)
        c2.metric("Predicted", pred, f"{100 * proba.max():.0f} % confidence")
        c3.markdown("<br>" + ('<span class="ok">✔ correct</span>' if pred == cls else '<span class="bad">✘ wrong</span>'),
                    unsafe_allow_html=True)
        c4.metric("Processing time", f"{(t2 - t0) * 1e3:.1f} ms", f"features {(t1 - t0) * 1e3:.1f} ms", delta_color="off")
        st.pyplot(plot_frame(x, info), use_container_width=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            order = np.argsort(proba)
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.barh([labels[i] for i in order], 100 * proba[order],
                    color=[CLASS_COLORS[labels[i]] for i in order])
            ax.set_xlim(0, 100); ax.set_xlabel("Probability (%)"); ax.set_title("Class probabilities")
            ax.grid(axis="x", alpha=.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with c2:
            imp = d["importance"]
            top = sorted(imp, key=imp.get, reverse=True)[:10]
            st.markdown("**Ten most important features — value for this frame**")
            st.dataframe(pd.DataFrame({"Feature": top, "Group": [R.FEATURE_GROUPS[k] for k in top],
                                       "Value": [feats[k] for k in top]}).set_index("Feature")
                         .style.format({"Value": "{:.3f}"}), use_container_width=True, height=390)
        if ints:
            st.caption("Interferers: " + ", ".join(f"{i['class']} ({i['type']}, SIR {i['sir']:.0f} dB, "
                                                   f"{i['fc'] / 1e6:+.1f} MHz)" for i in info["interferers"]))
    with tab_batch:
        st.markdown("Generate many random frames at a chosen impairment level and measure the accuracy live.")
        c1, c2, c3 = st.columns(3)
        lvl = c1.selectbox("Impairment level", list(R.LEVELS), format_func=lambda k: R.LEVELS[k], index=2)
        bsnr = c2.select_slider("SNR (dB)", ["random −10…30"] + list(range(-10, 31, 5)), value="random −10…30")
        nper = c3.slider("Frames per class", 5, 40, 15)
        if st.button("▶ Run batch test", type="primary"):
            rng = np.random.default_rng(int(seed) + 100)
            yt, yp = [], []
            bar = st.progress(0.0)
            tot = nper * len(R.CLASSES)
            for i, c in enumerate(R.CLASSES):
                for j in range(nper):
                    s = None if isinstance(bsnr, str) else bsnr
                    xx, _ = R.make_frame(rng, c, snr_db=s, level=lvl)
                    yt.append(c); yp.append(mdl.predict([R.features_vector(xx)[0]])[0])
                    bar.progress((i * nper + j + 1) / tot)
            bar.empty()
            yt, yp = np.array(yt), np.array(yp)
            acc = np.mean(yt == yp)
            st.metric("Accuracy", f"{100 * acc:.1f} %", f"{tot} frames")
            C = pd.crosstab(pd.Series(yt, name="True"), pd.Series(yp, name="Predicted"), normalize="index") * 100
            C = C.reindex(index=R.CLASSES, columns=R.CLASSES, fill_value=0)
            st.dataframe(C.style.format("{:.0f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)
            st.caption("Row-normalised confusion matrix (%).")
    with tab_model:
        st.markdown(f"""
- **Training data:** {d['n_train'] + d['n_test']} frames (220 per class) at random levels L1–L6, SNR −10…30 dB.
- **Model:** Random Forest, 200 trees, standardised features; trained in {d['train_time']:.1f} s.
- **Held-out accuracy:** {100 * d['test_acc']:.1f} % over all levels.
""")
        st.dataframe(pd.DataFrame({"Level": [R.LEVELS[k] for k in sorted(d["acc_by_level"])],
                                   "Accuracy (%)": [100 * d["acc_by_level"][k] for k in sorted(d["acc_by_level"])]})
                     .set_index("Level").style.format("{:.1f}"), use_container_width=True)
        st.caption("Same trend as the MATLAB study (84 % at L1 → 63 % at L6). The waveforms here are simplified, "
                   "so the reported numbers are those of the MATLAB run.")
        if st.button("Retrain the demo model"):
            if os.path.isfile(MODEL_FILE):
                os.remove(MODEL_FILE)
            get_model.clear()
            st.rerun()


PAGES = {"🏠 Overview": page_overview, "1️⃣ Project 1 · Deep learning": page_p1,
         "2️⃣ Project 2 · Machine learning": page_p2, "🔴 Live demo · Project 1": page_demo1,
         "🔴 Live demo · Project 2": page_demo}
with st.sidebar:
    st.markdown("## 📡 RF Classification")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
PAGES[page]()
