function [x, meta] = p2_makeFrame(cfg, pool, cls, level, opts)
%P2_MAKEFRAME  One received frame of class cls at impairment level 1-6.
%
%   [x, meta] = p2_makeFrame(cfg, pool, cls, level, opts)
%
%   Levels follow the handout's list, each keeping the ones before it:
%     1  different SNR levels (AWGN, signal at band centre)
%     2  + frequency offsets (random centre frequency + oscillator error)
%     3  + multipath fading
%     4  + one adjacent-channel interferer
%     5  + one interferer, adjacent or co-channel
%     6  + 1 to 3 simultaneously active interferers
%
%   The label is always the wanted signal, which is the strongest one
%   (signal-to-interference ratio cfg.sirRange, 0-20 dB).
%
%   opts (optional): .snr fixed in-band SNR (dB); .offsetRange [fmin fmax]
%
%   Noise has total power 1 over the band; a signal of in-band SNR "snr"
%   and bandwidth B has on-power 10^(snr/10)*B/fs.

if nargin < 5
    opts = struct();
end
N  = cfg.frameLen;
fs = cfg.fs;
cls = string(cls);
meta = struct("cls",cls, "level",level, "snr",NaN, "bw",NaN, ...
    "subtype","", "fc",NaN, "fading","none", "numInterf",0, ...
    "interfClasses","", "interfTypes","", "sirMin",NaN);
noise = sqrt(1/2)*(randn(N,1) + 1j*randn(N,1));
if cls == "Noise"
    x = noise;
    return
end

useOffset = level >= 2;
useFading = level >= 3;
offRange = [];
if isfield(opts, "offsetRange")
    offRange = opts.offsetRange;
    useOffset = true;
end
if isfield(opts, "snr")
    snr = opts.snr;
else
    snr = cfg.snrRange(1) + rand*diff(cfg.snrRange);
end

[seg, m, bw, sub] = p2_drawSegment(pool.(cls), N, cfg);
pOn = 10^(snr/10)*bw/fs;
[x, fc, fadName] = p2_conditionSignal(seg, m, bw, pOn, cfg, useOffset, useFading, offRange);
meta.snr = snr;  meta.bw = bw;  meta.subtype = sub;
meta.fc = fc;    meta.fading = fadName;

% ------------------------------------------------------- interferers
types = strings(0,1);
kinds = ["adjacent" "cochannel"];
if level == 4
    types = "adjacent";
elseif level == 5
    types = kinds(randi(2));
elseif level >= 6
    types = kinds(randi(2, randi(cfg.maxInterferers), 1)).';
end
others = setdiff(cfg.signalClasses, cls);
iCls = strings(0,1);
iType = strings(0,1);
sirs = zeros(0,1);
for t = types(:).'
    for attempt = 1:6
        ic = others(randi(numel(others)));
        [segI, mI, bwI] = p2_drawSegment(pool.(ic), N, cfg);
        fI = localInterfererFreq(fc, bw, bwI, t, cfg);
        if isnan(fI)
            continue
        end
        sir = cfg.sirRange(1) + rand*diff(cfg.sirRange);
        sI = p2_conditionSignal(segI, mI, bwI, pOn/10^(sir/10), cfg, false, useFading);
        x = x + sI.*exp(1j*2*pi*fI*(0:N-1)'/fs);
        iCls(end+1,1) = ic;   %#ok<AGROW>
        iType(end+1,1) = t;   %#ok<AGROW>
        sirs(end+1,1) = sir;  %#ok<AGROW>
        break
    end
end
x = x + noise;
meta.numInterf = numel(iCls);
if ~isempty(iCls)
    meta.interfClasses = strjoin(iCls, "+");
    meta.interfTypes   = strjoin(unique(iType), "+");
    meta.sirMin        = min(sirs);
end
end

% -------------------------------------------------------------------------
function f = localInterfererFreq(fc, bw, bwI, type, cfg)
% Centre frequency for an interferer next to (adjacent) or on top of
% (co-channel) the wanted signal at fc. NaN if it does not fit.
lim = cfg.fs/2*cfg.bandEdgeUse;
if type == "cochannel"
    f = fc + (2*rand - 1)*bw/2;
    f = min(max(f, -lim), lim);
    return
end
minSep = (bw + bwI)/2 + cfg.aciGuard;
sides = [-1 1];
sides = sides(randperm(2));
for sd = sides
    f = fc + sd*(minSep + rand*3e6);
    if abs(f) <= lim
        return
    end
end
f = NaN;
end
