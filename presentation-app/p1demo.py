"""Live demo for Project 1: pixel-wise labelling of a Bluetooth + WiFi spectrogram.

Project 1 used a deep semantic-segmentation network in MATLAB. That network
cannot run in Python, so this demo shows the same task with a lighter model:
every spectrogram pixel is described by a few local time-frequency features
and a gradient-boosted tree classifier labels it Noise, Bluetooth or WLAN.
The frame layout follows Project 1 (wideband capture around 2.44 GHz,
Bluetooth packets in 625 us slots hopping over 1 MHz channels, WiFi packets
on a 20 MHz channel, in-band SNR 5-40 dB).
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy import ndimage
from scipy import signal as sps

FS = 61.44e6            # 61.44 MHz complex sampling: a 60 MHz wide view
T = 4e-3                # 4 ms frame
NS = int(FS * T)
NFFT = 256              # 240 kHz frequency bins, 4.2 us time slices
LABELS = ["Noise", "Bluetooth", "WLAN"]
BT_BW, WLAN_BW = 1.0e6, 16.6e6


def _qpsk(rng, n):
    return ((rng.integers(0, 2, n) * 2 - 1) + 1j * (rng.integers(0, 2, n) * 2 - 1)) / np.sqrt(2)


def _bt_packet(rng, nbits):
    """GFSK, 1 Msym/s, BT = 0.5, h = 0.32, generated at 8 MHz and resampled to 61.44 MHz."""
    sps8 = 8
    g = sps.windows.gaussian(4 * sps8, std=0.375 * sps8)
    g /= g.sum()
    bits = rng.integers(0, 2, nbits) * 2 - 1
    f = np.convolve(np.repeat(bits, sps8), g, "same")
    x = np.exp(1j * np.cumsum(np.pi * 0.32 * f / sps8))
    return sps.resample_poly(x, 192, 25)          # 8 -> 61.44 MHz


def _wlan_packet(rng, nsym):
    """802.11a/g-like OFDM packet, 64-point FFT at 20 MHz, resampled to 61.44 MHz."""
    used = np.r_[-26:0, 1:27] % 64
    out = []
    for _ in range(nsym + 4):
        X = np.zeros(64, complex)
        X[used] = _qpsk(rng, 52)
        t = np.fft.ifft(X) * 8
        out.append(np.r_[t[-16:], t])
    return sps.resample_poly(np.concatenate(out), 384, 125)  # 20 -> 61.44 MHz


def make_frame(rng, content="Both", snr_bt=None, snr_wlan=None, overlap=False, wlan_fc=None):
    """One frame. content: 'Noise', 'Bluetooth', 'WLAN' or 'Both'.

    Returns x (complex samples) and the list of transmissions
    (class, start sample, end sample, centre frequency, bandwidth).
    """
    x = (rng.standard_normal(NS) + 1j * rng.standard_normal(NS)) / np.sqrt(2)
    tx = []
    lim = FS / 2 * 0.92
    if wlan_fc is None:
        wlan_fc = rng.uniform(-lim + WLAN_BW / 2, lim - WLAN_BW / 2)
    if content in ("WLAN", "Both"):
        snr = rng.uniform(5, 40) if snr_wlan is None else snr_wlan
        amp = np.sqrt(10 ** (snr / 10) * WLAN_BW / FS)
        s = int(rng.integers(0, NS // 5))
        while s < NS:
            pkt = _wlan_packet(rng, int(rng.integers(20, 160)))
            e = min(s + len(pkt), NS)
            ph = np.exp(2j * np.pi * wlan_fc * np.arange(s, e) / FS)
            x[s:e] += amp * pkt[:e - s] * ph
            tx.append(("WLAN", s, e, wlan_fc, WLAN_BW))
            s = e + int(rng.uniform(50e-6, 600e-6) * FS)
    if content in ("Bluetooth", "Both"):
        snr = rng.uniform(5, 40) if snr_bt is None else snr_bt
        amp = np.sqrt(10 ** (snr / 10) * BT_BW / FS)
        slot = int(625e-6 * FS)
        for s in range(int(rng.integers(0, slot)) - slot, NS, slot):
            if rng.random() > 0.75:
                continue                                   # idle slot
            # hop to a 1 MHz channel; without "overlap" keep clear of the WiFi channel
            for _ in range(50):
                fc = np.round(rng.uniform(-lim + 1e6, lim - 1e6) / 1e6) * 1e6
                if overlap or content != "Both" or abs(fc - wlan_fc) > WLAN_BW / 2 + 1e6:
                    break
            pkt = _bt_packet(rng, 366)
            a, b = max(s, 0), min(s + len(pkt), NS)
            if b <= a:
                continue
            ph = np.exp(2j * np.pi * fc * np.arange(a, b) / FS)
            x[a:b] += amp * pkt[a - s:b - s] * ph
            tx.append(("Bluetooth", a, b, fc, BT_BW))
    return x, tx


def spectrogram(x):
    f, t, S = sps.stft(x, fs=FS, window="hann", nperseg=NFFT, noverlap=0, return_onesided=False,
                       boundary=None, padded=False)
    S = np.abs(np.fft.fftshift(S, axes=0)) ** 2
    return np.fft.fftshift(f), t, S


def true_labels(tx, f, t):
    """Pixel label map (freq x time): 0 noise, 1 Bluetooth, 2 WLAN (Bluetooth drawn on top)."""
    L = np.zeros((len(f), len(t)), np.int8)
    dt = NFFT / FS
    for cls in ("WLAN", "Bluetooth"):
        for c, a, b, fc, bw in tx:
            if c != cls:
                continue
            tt = (t >= a / FS + 0.25 * dt) & (t <= b / FS - 0.25 * dt)
            ff = np.abs(f - fc) <= bw / 2
            L[np.ix_(ff, tt)] = 1 if cls == "Bluetooth" else 2
    return L


FEATURE_NAMES = ["power dB", "mean 3x3", "freq mean 5", "freq mean 17", "freq mean 65", "freq std 17",
                 "time mean 9", "time mean 41", "local max 5x5", "local std 5x5", "row persistence",
                 "edge 17 vs 65"]


def pixel_features(S):
    """Local time-frequency features of every pixel (freq x time x features)."""
    floor = np.percentile(S, 20) / 0.2231
    D = 10 * np.log10(S / floor + 1e-12)
    u = ndimage.uniform_filter
    F = [D,
         u(D, (3, 3)),
         u(D, (5, 1)), u(D, (17, 1)), u(D, (65, 1)),
         np.sqrt(np.maximum(u(D ** 2, (17, 1)) - u(D, (17, 1)) ** 2, 0)),
         u(D, (1, 9)), u(D, (1, 41)),
         ndimage.maximum_filter(D, (5, 5)),
         np.sqrt(np.maximum(u(D ** 2, (5, 5)) - u(D, (5, 5)) ** 2, 0)),
         np.repeat(np.mean(D > 6, axis=1, keepdims=True), D.shape[1], axis=1),
         u(D, (17, 1)) - u(D, (65, 1))]
    return np.stack(F, axis=-1).astype(np.float32)


def predict_map(model, S):
    F = pixel_features(S)
    shp = F.shape[:2]
    y = model.predict(F.reshape(-1, F.shape[-1]))
    return y.reshape(shp).astype(np.int8)


def metrics(Lt, Lp):
    acc = float(np.mean(Lt == Lp))
    iou = {}
    for k, name in enumerate(LABELS):
        inter = np.sum((Lt == k) & (Lp == k))
        union = np.sum((Lt == k) | (Lp == k))
        iou[name] = float(inter / union) if union else float("nan")
    return acc, iou


def build_and_train(n_frames=90, pix_per_frame=6000, seed=11, progress=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    contents = ["Bluetooth", "WLAN", "Both", "Both", "Noise"]
    for i in range(n_frames):
        x, tx = make_frame(rng, contents[i % len(contents)], overlap=rng.random() < 0.3)
        f, t, S = spectrogram(x)
        L = true_labels(tx, f, t).ravel()
        F = pixel_features(S).reshape(-1, len(FEATURE_NAMES))
        # balanced sampling: signal pixels are rare (Bluetooth < 1 %)
        idx = []
        for k in range(3):
            pool = np.nonzero(L == k)[0]
            if len(pool):
                idx.append(rng.choice(pool, min(len(pool), pix_per_frame // 3), replace=False))
        idx = np.concatenate(idx)
        Xs.append(F[idx]); ys.append(L[idx])
        if progress:
            progress((i + 1) / n_frames)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    t0 = time.time()
    mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                         random_state=1)
    mdl.fit(X, y)
    return mdl, time.time() - t0, len(y)


def evaluate(model, n=15, seed=99):
    rng = np.random.default_rng(seed)
    acc, ious = [], {k: [] for k in LABELS}
    for i in range(n):
        x, tx = make_frame(rng, ["Bluetooth", "WLAN", "Both"][i % 3])
        f, t, S = spectrogram(x)
        a, io = metrics(true_labels(tx, f, t), predict_map(model, S))
        acc.append(a)
        for k in LABELS:
            if not np.isnan(io[k]):
                ious[k].append(io[k])
    return float(np.mean(acc)), {k: float(np.mean(v)) for k, v in ious.items()}


def load_or_train(path, progress=None):
    import joblib
    import sklearn
    key = f"p1v1-{sklearn.__version__}-{len(FEATURE_NAMES)}"
    if os.path.isfile(path):
        try:
            d = joblib.load(path)
            if d.get("key") == key:
                return d
        except Exception:
            pass
    t0 = time.time()
    mdl, ttrain, npix = build_and_train(progress=progress)
    acc, iou = evaluate(mdl)
    d = {"key": key, "model": mdl, "train_time": ttrain, "n_pixels": npix, "test_acc": acc,
         "test_iou": iou, "build_time": time.time() - t0}
    try:
        joblib.dump(d, path)
    except Exception:
        pass
    return d
