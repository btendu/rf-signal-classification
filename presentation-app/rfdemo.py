"""Signal generation, impairments, feature extraction and model for the live demo.

This is a compact Python re-implementation of the approach of Project 2:
the same seven classes, the same frame (8192 samples at 30.72 MHz), the
same kind of impairments and the same four feature groups. The waveforms
are simplified (correct numerology, bandwidth and timing, random payload
bits), so the demo shows the method live; the reported results come from
the MATLAB run.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy import signal as sps

FS = 30.72e6
N = 8192
CLASSES = ["Noise", "WLAN", "Bluetooth", "ZigBee", "LTE", "NR5G", "Radar"]
SIGNALS = CLASSES[1:]
LEVELS = {
    1: "L1: different SNR levels",
    2: "L2: + frequency offset",
    3: "L3: + multipath fading",
    4: "L4: + adjacent-channel interference",
    5: "L5: + co-channel interference",
    6: "L6: + multiple simultaneous signals",
}


# ---------------------------------------------------------------- helpers
def _resample(x, up, down):
    return sps.resample_poly(x, up, down).astype(np.complex128)


def _fit(x, n):
    """Repeat or cut to n samples."""
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def _qam(rng, n, order):
    m = int(np.sqrt(order))
    lv = np.arange(m) * 2 - (m - 1)
    s = rng.choice(lv, n) + 1j * rng.choice(lv, n)
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


def _window(rng, x, m, n):
    """Random window of n samples that contains enough of the signal."""
    duty = m.mean()
    need = min(0.3, 0.8 * duty)
    best, bs = -1, 0
    for _ in range(60):
        s = int(rng.integers(0, len(x) - n))
        a = m[s:s + n].mean()
        if a >= need:
            return x[s:s + n], m[s:s + n]
        if a > best:
            best, bs = a, s
    return x[bs:bs + n], m[bs:bs + n]


# ---------------------------------------------------------------- generators
# Each generator returns (x, on, bw): samples at FS, on/off mask, occupied bandwidth (Hz).
def gen_wlan(rng, n_total):
    """802.11a/g-like OFDM packets: 64-pt FFT at 20 MHz, 52 used subcarriers."""
    used = np.r_[-26:0, 1:27]
    out, on = [], []
    while sum(len(o) for o in out) < 2 * n_total:
        nsym = rng.integers(10, 80)
        order = rng.choice([4, 16, 64])
        sym = []
        for k in range(nsym + 4):                      # 4 symbols ~ preamble
            X = np.zeros(64, complex)
            X[used % 64] = _qam(rng, 52, 4 if k < 4 else order)
            t = np.fft.ifft(X) * 8
            sym.append(np.r_[t[-16:], t])
        pkt = _resample(np.concatenate(sym), 192, 125)  # 20 MHz -> 30.72 MHz
        gap = int(rng.uniform(16e-6, 120e-6) * FS)
        out += [pkt, np.zeros(gap)]
        on += [np.ones(len(pkt)), np.zeros(gap)]
    x, m = _window(rng, np.concatenate(out), np.concatenate(on), n_total)
    return x, m, 16.6e6


def gen_bluetooth(rng, n_total):
    """GFSK, 1 Msym/s, BT = 0.5, h = 0.32, packets of 366 µs to 2.9 ms."""
    sps8 = 8
    g = sps.windows.gaussian(4 * sps8, std=0.375 * sps8)     # BT = 0.5
    g /= g.sum()
    out, on = [], []
    while sum(len(o) for o in out) < 2 * n_total:
        nb = rng.choice([366, 1622, 2870])
        bits = rng.integers(0, 2, nb) * 2 - 1
        f = np.convolve(np.repeat(bits, sps8), g, "same")
        ph = np.cumsum(np.pi * 0.32 * f / sps8)
        pkt = _resample(np.exp(1j * ph), 96, 25)        # 8 MHz -> 30.72 MHz
        gap = int(rng.uniform(100e-6, 600e-6) * FS)
        out += [np.zeros(gap // 2), pkt, np.zeros(gap - gap // 2)]
        on += [np.zeros(gap // 2), np.ones(len(pkt)), np.zeros(gap - gap // 2)]
    x, m = _window(rng, np.concatenate(out), np.concatenate(on), n_total)
    return x, m, 1.0e6


def gen_zigbee(rng, n_total):
    """802.15.4 O-QPSK with half-sine pulses, 2 Mchip/s."""
    spc = 4                                              # 8 MHz, 4 samples per chip
    out, on = [], []
    while sum(len(o) for o in out) < 2 * n_total:
        nch = rng.integers(400, 3000) // 2 * 2
        c = rng.integers(0, 2, nch) * 2 - 1
        p = np.sin(np.pi * np.arange(2 * spc) / (2 * spc))
        i = np.zeros(nch // 2 * 2 * spc + spc)
        q = np.zeros_like(i)
        for k in range(nch // 2):
            i[k * 2 * spc:(k + 1) * 2 * spc] += c[2 * k] * p
            q[k * 2 * spc + spc:(k + 1) * 2 * spc + spc] += c[2 * k + 1] * p
        pkt = _resample(i + 1j * q, 96, 25)
        gap = int(rng.uniform(150e-6, 800e-6) * FS)
        out += [pkt, np.zeros(gap)]
        on += [np.ones(len(pkt)), np.zeros(gap)]
    x, m = _window(rng, np.concatenate(out), np.concatenate(on), n_total)
    return x, m, 2.0e6


def _ofdm_continuous(rng, n_total, nfft, nsc, cp):
    """Continuous CP-OFDM at FS with nsc used subcarriers around DC."""
    k = np.r_[-(nsc // 2):0, 1:nsc - nsc // 2 + 1]
    out = []
    while sum(len(o) for o in out) < n_total + nfft + cp:
        X = np.zeros(nfft, complex)
        X[k % nfft] = _qam(rng, len(k), rng.choice([4, 16, 64]))
        t = np.fft.ifft(X) * np.sqrt(nfft)
        out.append(np.r_[t[-cp:], t])
    x = np.concatenate(out)
    s = rng.integers(0, nfft)
    return x[s:s + n_total]


def gen_lte(rng, n_total):
    """LTE downlink numerology: 15 kHz, 2048-pt FFT, 25/50/75/100 RB."""
    nrb = rng.choice([25, 50, 75, 100])
    x = _ofdm_continuous(rng, n_total, 2048, 12 * nrb, 144)
    return x, np.ones(n_total), 12 * nrb * 15e3


def gen_nr(rng, n_total):
    """5G NR numerology: 15 kHz (2048-pt) or 30 kHz (1024-pt) subcarriers."""
    scs, nrb = [(15e3, 25), (15e3, 52), (15e3, 106), (30e3, 24), (30e3, 51)][rng.integers(5)]
    nfft = int(FS / scs)
    x = _ofdm_continuous(rng, n_total, nfft, 12 * nrb, 144 if nfft == 2048 else 72)
    return x, np.ones(n_total), 12 * nrb * scs


def gen_radar(rng, n_total):
    """Pulsed radar: LFM chirps or Barker-13 phase-coded pulses."""
    bw = rng.uniform(1e6, 20e6)
    pw = rng.uniform(5e-6, 40e-6)
    duty = rng.uniform(0.05, 0.25)
    npw = int(pw * FS)
    pri = int(npw / duty)
    t = np.arange(npw) / FS
    if rng.random() < 0.7:
        sgn = rng.choice([-1, 1])
        pulse = np.exp(1j * np.pi * sgn * bw / pw * (t - pw / 2) ** 2)
    else:
        b13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        pulse = b13[np.minimum((t / pw * 13).astype(int), 12)].astype(complex)
        bw = 13 / pw
    L = max(4 * n_total, 4 * pri)
    x = np.zeros(L + pri, complex)
    m = np.zeros(L + pri)
    for s in range(0, L, pri):
        j = s + int(rng.integers(0, pri // 50 + 1))
        b = min(j + npw, len(x))
        x[j:b] = pulse[:b - j]
        m[j:b] = 1
    x, m = _window(rng, x[:L], m[:L], n_total)
    return x, m, min(bw, 20e6)


GENERATORS = {"WLAN": gen_wlan, "Bluetooth": gen_bluetooth, "ZigBee": gen_zigbee,
              "LTE": gen_lte, "NR5G": gen_nr, "Radar": gen_radar}


# ---------------------------------------------------------------- impairments
def fade(rng, x):
    """Random 1-4 tap multipath channel with Rayleigh/Rician taps."""
    ntap = rng.integers(1, 5)
    delays = np.sort(rng.integers(0, 40, ntap))
    delays[0] = 0
    p = np.exp(-delays / 15.0)
    h = np.zeros(delays[-1] + 1, complex)
    k = rng.uniform(0, 10)
    for d, pw in zip(delays, p):
        h[d] += np.sqrt(pw / 2) * (rng.standard_normal() + 1j * rng.standard_normal())
    h[0] += np.sqrt(k)
    h /= np.sqrt(np.sum(np.abs(h) ** 2))
    return np.convolve(x, h)[:len(x)]


def condition(rng, raw, snr_db, f0, multipath, n=N):
    """Scale a generated signal to in-band SNR snr_db, fade it and move it to f0."""
    x, on, bw = raw
    p_on = np.mean(np.abs(x[on > 0]) ** 2) if on.any() else 1.0
    x = x / np.sqrt(p_on) * np.sqrt(10 ** (snr_db / 10) * bw / FS)
    if multipath:
        x = fade(rng, x)
    cfo = rng.uniform(-20e3, 20e3)
    return x * np.exp(2j * np.pi * (f0 + cfo) * np.arange(n) / FS)


def make_frame(rng, cls, snr_db=None, level=1, offset_hz=None, interferers=None, sir_db=None,
               multipath=None):
    """Build one frame.

    level sets the defaults (frequency offset from L2, fading from L3,
    interferers from L4); every argument given explicitly overrides them.
    interferers: list of (class, "adjacent" | "co-channel").
    Returns x and a dict describing the frame.
    """
    if snr_db is None:
        snr_db = rng.uniform(-10, 30)
    if multipath is None:
        multipath = level >= 3
    if interferers is None:
        interferers = []
        others = [c for c in SIGNALS if c != cls]
        if level == 4:
            interferers = [(str(rng.choice(others)), "adjacent")]
        elif level == 5:
            interferers = [(str(rng.choice(others)), str(rng.choice(["adjacent", "co-channel"])))]
        elif level == 6:
            for _ in range(rng.integers(1, 4)):
                interferers.append((str(rng.choice(others)), str(rng.choice(["adjacent", "co-channel"]))))
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    info = {"class": cls, "snr": float(snr_db), "level": level, "interferers": [], "fc": 0.0,
            "bw": 0.0, "multipath": bool(multipath)}
    if cls == "Noise":
        return x, info
    lim = FS / 2 * 0.95
    raw = GENERATORS[cls](rng, N)
    bw = raw[2]
    if offset_hz is None:
        offset_hz = rng.uniform(-1, 1) * max(lim - bw / 2, 0) if level >= 2 else 0.0
    fc = float(np.clip(offset_hz, -lim + bw / 2, lim - bw / 2)) if bw < 2 * lim else 0.0
    x = x + condition(rng, raw, snr_db, fc, multipath)
    p_wanted = 10 ** (snr_db / 10) * bw / FS
    for icls, kind in interferers:
        sir = float(rng.uniform(0, 20) if sir_db is None else sir_db)
        iraw = GENERATORS[icls](rng, N)
        ibw = iraw[2]
        if kind == "adjacent":
            sep = (bw + ibw) / 2 + 0.5e6 + rng.uniform(0, 3e6)
            sgn = rng.choice([-1, 1])
            fi = fc + sgn * sep
            if abs(fi) > lim - ibw / 2:
                fi = fc - sgn * sep
        else:
            fi = fc + rng.uniform(-0.5, 0.5) * bw
        fi = float(np.clip(fi, -lim + ibw / 2, lim - ibw / 2)) if ibw < 2 * lim else 0.0
        p_int = p_wanted / 10 ** (sir / 10)
        x = x + condition(rng, iraw, 10 * np.log10(p_int * FS / ibw), fi, multipath)
        info["interferers"].append({"class": icls, "type": kind, "sir": sir, "fc": fi})
    info.update({"fc": fc, "bw": bw})
    return x, info


# ---------------------------------------------------------------- features
FEATURE_GROUPS = {}


def _skew(v):
    v = v - v.mean()
    return float(np.mean(v ** 3) / (np.mean(v ** 2) ** 1.5 + 1e-16))


def _kurt(v):
    v = v - v.mean()
    return float(np.mean(v ** 4) / (np.mean(v ** 2) ** 2 + 1e-16))


def extract_features(x, nfft=512, n2=128):
    """Handcrafted features in the four groups of the handout."""
    f = {}

    def put(group, name, val):
        FEATURE_GROUPS[name] = group
        f[name] = float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))

    x = x.astype(complex)
    praw = np.mean(np.abs(x) ** 2) + 1e-16
    xc = x - x.mean()
    xn = xc / np.sqrt(np.mean(np.abs(xc) ** 2) + 1e-16)
    a = np.abs(xn)

    # spectra first (noise floor is needed by the time features)
    w = sps.windows.hann(nfft, sym=False)
    fr, P = sps.welch(x, fs=1.0, window=w, nperseg=nfft, noverlap=nfft // 2, return_onesided=False,
                      detrend=False, scaling="density")
    P = np.fft.fftshift(P) + 1e-20
    fr = np.fft.fftshift(fr)
    floor = np.sort(P)[max(0, int(round(0.1 * nfft)) - 1)]
    pn = P / P.sum()
    pdb = 10 * np.log10(P)

    # time domain
    put("Time", "meanEnvelope", a.mean())
    put("Time", "varEnvelope", a.var())
    put("Time", "rmsDbAboveNoise", 10 * np.log10(praw / (floor * 1.0 + 1e-20)))
    put("Time", "paprDb", 10 * np.log10(np.max(a ** 2) / np.mean(a ** 2)))
    put("Time", "skewEnvelope", _skew(a))
    put("Time", "kurtEnvelope", _kurt(a))
    put("Time", "kurtI", _kurt(xn.real))
    put("Time", "zcrI", np.mean(np.diff(np.sign(xn.real)) != 0))

    # frequency domain
    c = np.sum(fr * pn)
    sd = np.sqrt(np.sum((fr - c) ** 2 * pn))
    cs = np.cumsum(pn)
    put("Frequency", "spectralCentroid", c)
    put("Frequency", "spectralBandwidth", sd)
    put("Frequency", "spectralFlatness", np.exp(np.mean(np.log(P))) / P.mean())
    put("Frequency", "spectralEntropy", -np.sum(pn * np.log(pn + 1e-20)) / np.log(nfft))
    put("Frequency", "obw99", (np.argmax(cs >= 0.995) - np.argmax(cs >= 0.005) + 1) / nfft)
    put("Frequency", "bwAboveFloor6dB", np.mean(P > 4 * floor))
    put("Frequency", "peakToAvgPsdDb", 10 * np.log10(P.max() / P.mean()))
    put("Frequency", "spectralSkewness", np.sum((fr - c) ** 3 * pn) / (sd ** 3 + 1e-16))
    put("Frequency", "spectralKurtosis", np.sum((fr - c) ** 4 * pn) / (sd ** 4 + 1e-16))
    pk, _ = sps.find_peaks(pdb, prominence=6)
    put("Frequency", "numPsdPeaks", len(pk))
    put("Frequency", "psdMaxMinusMedianDb", pdb.max() - np.median(pdb))

    # higher-order statistics
    m20 = np.mean(xn ** 2)
    m21 = np.mean(a ** 2)
    m40 = np.mean(xn ** 4)
    m41 = np.mean(xn ** 3 * np.conj(xn))
    m42 = np.mean(a ** 4)
    m60 = np.mean(xn ** 6)
    m63 = np.mean(a ** 6)
    put("Higher-order", "absC20", abs(m20))
    put("Higher-order", "absC40", abs(m40 - 3 * m20 ** 2))
    put("Higher-order", "absC41", abs(m41 - 3 * m20 * m21))
    put("Higher-order", "C42", np.real(m42 - abs(m20) ** 2 - 2 * m21 ** 2))
    put("Higher-order", "absC60", abs(m60 - 15 * m20 * m40 + 30 * m20 ** 3))
    put("Higher-order", "C63", np.real(m63 - 9 * m21 * m42 + 12 * m21 ** 3))
    put("Higher-order", "envelopeCV", a.std() / (a.mean() + 1e-16))
    strong = a > 0.5 * np.median(a)
    ph = np.unwrap(np.angle(xn))
    nl = ph - np.polyval(np.polyfit(np.arange(N), ph, 1), np.arange(N))
    put("Higher-order", "sigmaAbsPhase", np.std(np.abs(nl[strong])) if strong.any() else 0)
    inst = np.diff(ph)
    put("Higher-order", "stdInstFreq", inst.std())
    put("Higher-order", "kurtInstFreq", _kurt(inst))

    # time-frequency
    _, _, S = sps.stft(x, fs=1.0, window="hann", nperseg=n2, noverlap=n2 // 2, return_onesided=False,
                       boundary=None, padded=False)
    S = np.abs(np.fft.fftshift(S, axes=0)) ** 2
    sv = np.sort(S.ravel())
    noise_mean = sv[max(0, int(round(0.2 * sv.size)) - 1)] / 0.2231
    slice_p = S.sum(0)
    active = slice_p > 1.5 * n2 * noise_mean
    sel = active if active.sum() >= 2 else np.ones(S.shape[1], bool)
    fb = (np.arange(n2) - n2 / 2) / n2
    Sn = S / (S.sum(0, keepdims=True) + 1e-20)
    cent = (fb[:, None] * Sn).sum(0)
    bwt = np.sqrt(((fb[:, None] - cent) ** 2 * Sn).sum(0))
    peak = fb[S.argmax(0)]
    put("Time-frequency", "timeOccupancy", active.mean())
    put("Time-frequency", "freqOccupancy", np.mean(S.mean(1) > 4 * noise_mean))
    put("Time-frequency", "tfOccupancy", np.mean(S > 4 * noise_mean))
    put("Time-frequency", "stdSliceCentroid", cent[sel].std())
    put("Time-frequency", "meanSliceBandwidth", bwt[sel].mean())
    put("Time-frequency", "stdSliceBandwidth", bwt[sel].std())
    flux = np.sum(np.diff(Sn, axis=1) ** 2, 0)
    put("Time-frequency", "maxSpectralFlux", flux.max() if flux.size else 0)
    put("Time-frequency", "peakHopRate", np.mean(np.abs(np.diff(peak[sel])) > 2 / n2) if sel.sum() > 1 else 0)
    tp = S / S.sum()
    put("Time-frequency", "tfEntropy", -np.sum(tp * np.log(tp + 1e-20)) / np.log(tp.size))
    put("Time-frequency", "slicePowerCV", slice_p.std() / (slice_p.mean() + 1e-16))
    # chirp: straight-line fit of the peak frequency inside each burst of active slices
    r2, slope, wsum = 0.0, 0.0, 0.0
    runs, start = [], None
    for i, v in enumerate(np.r_[active, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    for a0, a1 in runs:
        if a1 - a0 >= 3:
            ta = np.arange(a0, a1)
            pf = peak[a0:a1]
            cc = np.polyfit(ta, pf, 1)
            res = pf - np.polyval(cc, ta)
            sst = np.sum((pf - pf.mean()) ** 2)
            rr = 0.0 if sst < 1e-12 else max(0.0, 1 - np.sum(res ** 2) / sst)
            r2 += rr * (a1 - a0); slope += abs(cc[0]) * (a1 - a0); wsum += a1 - a0
    if wsum:
        r2, slope = r2 / wsum, slope / wsum
    put("Time-frequency", "chirpLinearityR2", r2)
    put("Time-frequency", "chirpSlope", slope * 1e3)
    return f


FEATURE_NAMES = list(extract_features(np.ones(N, complex) + 0.01 * np.random.default_rng(0).standard_normal(N)).keys())


def features_vector(x):
    f = extract_features(x)
    return np.array([f[k] for k in FEATURE_NAMES]), f


# ---------------------------------------------------------------- model
def build_dataset(n_per_class=220, seed=2026, levels=(1, 2, 3, 4, 5, 6), progress=None):
    rng = np.random.default_rng(seed)
    X, y, lv = [], [], []
    total = n_per_class * len(CLASSES)
    k = 0
    for cls in CLASSES:
        for _ in range(n_per_class):
            L = int(rng.choice(levels))
            x, _ = make_frame(rng, cls, level=L)
            X.append(features_vector(x)[0])
            y.append(cls)
            lv.append(L)
            k += 1
            if progress and k % 20 == 0:
                progress(k / total)
    return np.array(X), np.array(y), np.array(lv)


def train_model(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    mdl = make_pipeline(StandardScaler(),
                        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=1,
                                               class_weight="balanced"))
    mdl.fit(X, y)
    return mdl, time.time() - t0


def load_or_train(path, progress=None):
    """Load the cached model, or build the data set and train it (first start)."""
    import joblib
    import sklearn
    key = f"v3-{sklearn.__version__}-{len(FEATURE_NAMES)}"
    if os.path.isfile(path):
        try:
            d = joblib.load(path)
            if d.get("key") == key:
                return d
        except Exception:
            pass
    t0 = time.time()
    X, y, lv = build_dataset(progress=progress)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte, ltr, lte = train_test_split(X, y, lv, test_size=0.25, stratify=y, random_state=1)
    mdl, ttrain = train_model(Xtr, ytr)
    acc = float(np.mean(mdl.predict(Xte) == yte))
    acc_lv = {int(L): float(np.mean(mdl.predict(Xte[lte == L]) == yte[lte == L])) for L in np.unique(lte)}
    imp = mdl[-1].feature_importances_
    d = {"key": key, "model": mdl, "test_acc": acc, "acc_by_level": acc_lv, "train_time": ttrain,
         "build_time": time.time() - t0, "n_train": len(ytr), "n_test": len(yte),
         "importance": dict(zip(FEATURE_NAMES, imp.tolist()))}
    try:
        joblib.dump(d, path)
    except Exception:
        pass
    return d
