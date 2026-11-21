function [x, meta] = p2_makeCoexFrame(cfg, pool, A, B, kind)
%P2_MAKECOEXFRAME  One frame of a coexistence scenario "A vs B".
%
%   [x, meta] = p2_makeCoexFrame(cfg, pool, A, B, kind)
%
%   kind = "A", "B", "AB" (both active) or "Noise".
%   The label is A, B, "A+B" or "Noise": the classifier must tell which of
%   the two systems are on the air, i.e. identify the interference.
%   Both signals get random centre frequencies, so they sometimes overlap
%   and sometimes not; multipath fading is applied (level-3 conditions).

N  = cfg.frameLen;
fs = cfg.fs;
A = string(A);  B = string(B);
meta = struct("pair",A+" vs "+B, "label","", "snrA",NaN, "snrB",NaN, ...
    "sir",NaN, "sep",NaN, "overlapRatio",NaN, "bwA",NaN, "bwB",NaN);
x = sqrt(1/2)*(randn(N,1) + 1j*randn(N,1));
r = cfg.coexSnrRange;

switch string(kind)
    case "Noise"
        meta.label = "Noise";
    case "A"
        [x, meta.snrA, ~, meta.bwA] = localAdd(x, cfg, pool, A, r(1)+rand*diff(r), N, fs);
        meta.label = A;
    case "B"
        [x, meta.snrB, ~, meta.bwB] = localAdd(x, cfg, pool, B, r(1)+rand*diff(r), N, fs);
        meta.label = B;
    otherwise
        snrA = r(1) + rand*diff(r);
        sir  = cfg.coexSirRange(1) + rand*diff(cfg.coexSirRange);
        [x, ~, fA, bwA, pOnA] = localAdd(x, cfg, pool, A, snrA, N, fs);
        [segB, mB, bwB] = p2_drawSegment(pool.(B), N, cfg);
        pOnB = pOnA/10^(sir/10);
        [sB, fB] = p2_conditionSignal(segB, mB, bwB, pOnB, cfg, true, true);
        x = x + sB;
        meta.snrA = snrA;
        meta.snrB = 10*log10(pOnB*fs/bwB);
        meta.sir = sir;
        meta.bwA = bwA;  meta.bwB = bwB;
        meta.sep = abs(fA - fB);
        meta.overlapRatio = meta.sep/((bwA + bwB)/2);   % < 1 means overlap
        meta.label = A + "+" + B;
end
end

% -------------------------------------------------------------------------
function [x, snr, f0, bw, pOn] = localAdd(x, cfg, pool, cls, snr, N, fs)
[seg, m, bw] = p2_drawSegment(pool.(cls), N, cfg);
pOn = 10^(snr/10)*bw/fs;
[s, f0] = p2_conditionSignal(seg, m, bw, pOn, cfg, true, true);
x = x + s;
end
