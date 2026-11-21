function [f, names, groups] = p2_extractFeatures(x, fs, psdNfft, stftNfft)
%P2_EXTRACTFEATURES  Handcrafted RF features of one frame (handout list).
%
%   [f, names, groups] = p2_extractFeatures(x, fs, psdNfft, stftNfft)
%
%   f       1-by-52 feature vector (finite; NaN/Inf replaced by 0)
%   names   feature names
%   groups  "Time", "Frequency", "Higher-order" or "Time-frequency"
%
%   Frequencies are expressed as fractions of fs and amplitudes are
%   normalised to unit RMS, so the features do not depend on receiver gain.
%   The same function is used for simulated frames and SDR captures.

if nargin < 3, psdNfft = 512; end
if nargin < 4, stftNfft = 128; end
x = double(x(:));
N = numel(x);
vals = zeros(1,60);
names = strings(1,60);
groups = strings(1,60);
k = 0;

% ---------------------------------------------------------- preparation
pRaw = mean(abs(x).^2) + eps;
mu   = mean(x);
xc   = x - mu;
xn   = xc/sqrt(mean(abs(xc).^2) + eps);       % zero mean, unit power
a    = abs(xn);                               % envelope

% Welch power spectral density (Hann, 50 % overlap), centred
n   = psdNfft;
w   = 0.5 - 0.5*cos(2*pi*(0:n-1)'/n);
hop = n/2;
M   = floor((N - n)/hop) + 1;
idx = (0:n-1)' + (0:M-1)*hop + 1;
X   = fftshift(fft(x(idx).*w, [], 1), 1);
P   = mean(abs(X).^2, 2)/(fs*sum(w.^2));      % W/Hz
fr  = ((0:n-1)' - n/2)/n;                     % frequency / fs
Ps  = sort(P);
floorP = Ps(max(1, round(0.10*n)));           % noise-floor estimate
pn  = P/sum(P);
PdB = 10*log10(P + eps);

% ------------------------------------------------------- time domain
put("meanEnvelope",     mean(a),                         "Time");
put("varEnvelope",      var(a),                          "Time");
put("dcRatio",          abs(mu)/sqrt(pRaw),              "Time");
put("rmsDbAboveNoise",  10*log10(pRaw/(floorP*fs + eps)), "Time");
put("paprDb",           10*log10(max(a.^2)/mean(a.^2)),  "Time");
put("skewEnvelope",     localSkew(a),                    "Time");
put("kurtEnvelope",     localKurt(a),                    "Time");
put("skewI",            localSkew(real(xn)),             "Time");
put("kurtI",            localKurt(real(xn)),             "Time");
put("skewQ",            localSkew(imag(xn)),             "Time");
put("kurtQ",            localKurt(imag(xn)),             "Time");
put("zeroCrossRateI",   mean(abs(diff(sign(real(xn)))) > 0), "Time");
put("zeroCrossRateQ",   mean(abs(diff(sign(imag(xn)))) > 0), "Time");

% -------------------------------------------------- frequency domain
c  = sum(fr.*pn);
sp = sqrt(sum((fr - c).^2.*pn)) + eps;
cs = cumsum(pn);
lo = find(cs >= 0.005, 1);
hi = find(cs >= 0.995, 1);
[pm, im] = max(P);
PdBs = movmean(PdB, 5);
try
    pk = findpeaks(PdBs, "MinPeakProminence", 6);
    nPeaks = numel(pk);
catch
    nPeaks = 0;
end
put("spectralCentroid",   c,                                  "Frequency");
put("spectralBandwidth",  sp,                                 "Frequency");
put("spectralVarianceDb", var(PdB),                           "Frequency");
put("spectralFlatness",   exp(mean(log(P + eps)))/mean(P),    "Frequency");
put("occupiedBw99",       (hi - lo + 1)/n,                    "Frequency");
put("bwAboveFloor6dB",    mean(P > 4*floorP),                 "Frequency");
put("peakFrequency",      fr(im),                             "Frequency");
put("peakToAvgPsdDb",     10*log10(pm/mean(P)),               "Frequency");
put("spectralSkewness",   sum(((fr - c)/sp).^3.*pn),          "Frequency");
put("spectralKurtosis",   sum(((fr - c)/sp).^4.*pn),          "Frequency");
put("spectralEntropy",    -sum(pn.*log(pn + eps))/log(n),     "Frequency");
put("numPsdPeaks",        nPeaks,                             "Frequency");
put("centrePowerFraction",sum(pn(abs(fr) <= 1/8)),            "Frequency");
put("psdMaxMinusMedianDb",max(PdB) - median(PdB),             "Frequency");

% --------------------------------------------- higher-order statistics
M20 = mean(xn.^2);
M21 = mean(a.^2);
M40 = mean(xn.^4);
M41 = mean(xn.^3.*conj(xn));
M42 = mean(a.^4);
M60 = mean(xn.^6);
M63 = mean(a.^6);
C40 = M40 - 3*M20^2;
C41 = M41 - 3*M20*M21;
C42 = M42 - abs(M20)^2 - 2*M21^2;
C60 = M60 - 15*M20*M40 + 30*M20^3;
C63 = M63 - 9*M21*M42 + 12*M21^3;              % circular-signal form

strong = a > 1;
if nnz(strong) < 16
    strong = true(N,1);
end
ph   = unwrap(angle(xn));
t    = (0:N-1)'/N;
pc   = polyfit(t, ph, 1);
phNL = angle(exp(1j*(ph - polyval(pc, t))));
fi   = diff(ph)/(2*pi);                        % cycles per sample
both = strong(1:end-1) & strong(2:end);
if nnz(both) < 16
    both = true(N-1,1);
end
put("moment42",        M42,              "Higher-order");
put("absC20",          abs(M20),         "Higher-order");
put("absC40",          abs(C40),         "Higher-order");
put("absC41",          abs(C41),         "Higher-order");
put("C42",             real(C42),        "Higher-order");
put("absC60",          abs(C60),         "Higher-order");
put("C63",             real(C63),        "Higher-order");
put("envelopeCV",      std(a)/(mean(a) + eps), "Higher-order");
put("sigmaDirectPhase",std(phNL(strong)),      "Higher-order");
put("sigmaAbsPhase",   std(abs(phNL(strong))), "Higher-order");
put("stdInstFreq",     std(fi(both)),          "Higher-order");
put("kurtInstFreq",    localKurt(fi(both)),    "Higher-order");

% ------------------------------------------------------ time-frequency
n2  = stftNfft;
w2  = 0.5 - 0.5*cos(2*pi*(0:n2-1)'/n2);
hop2 = n2/2;
M2  = floor((N - n2)/hop2) + 1;
idx2 = (0:n2-1)' + (0:M2-1)*hop2 + 1;
S   = abs(fftshift(fft(x(idx2).*w2, [], 1), 1)).^2;   % n2-by-M2
Sv  = sort(S(:));
noiseMean = Sv(max(1, round(0.2*numel(Sv))))/0.2231;  % exponential noise
sliceP = sum(S, 1);
active = sliceP > 1.5*n2*noiseMean;
if nnz(active) < 2
    sel = true(1, M2);
else
    sel = active;
end
fb  = ((0:n2-1)' - n2/2)/n2;
Sn  = S./(sum(S, 1) + eps);
cen = sum(fb.*Sn, 1);
bwS = sqrt(sum((fb - cen).^2.*Sn, 1));
flux = sum(diff(Sn, 1, 2).^2, 1);
[~, pkBin] = max(S, [], 1);
pkA = pkBin(sel);
tA  = find(sel);
if numel(pkA) >= 3
    cc  = polyfit(tA, fb(pkA)', 1);
    res = fb(pkA)' - polyval(cc, tA);
    sst = sum((fb(pkA)' - mean(fb(pkA))).^2);
    chirpR2 = max(0, 1 - sum(res.^2)/max(sst, eps));
    if sst < eps
        chirpR2 = 0;
    end
    chirpSlope = abs(cc(1))*M2;
else
    chirpR2 = 0;
    chirpSlope = 0;
end
q = S(:)/sum(S(:));
put("timeOccupancy",      mean(active),                       "Time-frequency");
put("freqOccupancy",      mean(mean(S, 2) > 2*noiseMean),     "Time-frequency");
put("tfOccupancy",        mean(S(:) > 4*noiseMean),           "Time-frequency");
put("stdSliceCentroid",   std(cen(sel)),                      "Time-frequency");
put("meanSliceBandwidth", mean(bwS(sel)),                     "Time-frequency");
put("stdSliceBandwidth",  std(bwS(sel)),                      "Time-frequency");
put("meanSpectralFlux",   mean(flux),                         "Time-frequency");
put("maxSpectralFlux",    max(flux),                          "Time-frequency");
put("peakHopRate",        mean(abs(diff(pkA)) > 2),           "Time-frequency");
put("chirpLinearityR2",   chirpR2,                            "Time-frequency");
put("chirpSlope",         chirpSlope,                         "Time-frequency");
put("tfEntropy",          -sum(q.*log(q + eps))/log(numel(q)), "Time-frequency");
put("slicePowerCV",       std(sliceP)/(mean(sliceP) + eps),   "Time-frequency");

vals   = vals(1:k);
names  = names(1:k);
groups = groups(1:k);
vals(~isfinite(vals)) = 0;
f = vals;

    function put(name, v, g)
        k = k + 1;
        vals(k) = double(real(v(1)));
        names(k) = name;
        groups(k) = g;
    end
end

% -------------------------------------------------------------------------
function s = localSkew(v)
v = v(:) - mean(v);
s = mean(v.^3)/(mean(v.^2)^1.5 + eps);
end

function k = localKurt(v)
v = v(:) - mean(v);
k = mean(v.^4)/(mean(v.^2)^2 + eps);
end
