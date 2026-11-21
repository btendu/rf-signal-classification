function [s, f0, fadName] = p2_conditionSignal(seg, m, bw, pOn, cfg, useOffset, useFading, offRange)
%P2_CONDITIONSIGNAL  Scale, fade and frequency-shift one signal segment.
%
%   [s, f0, fadName] = p2_conditionSignal(seg, m, bw, pOn, cfg, ...
%                                         useOffset, useFading, offRange)
%
%   pOn       power while the signal is on (noise has total power 1)
%   useOffset random centre frequency inside the band + oscillator error
%   useFading random multipath channel (p2_applyFading)
%   offRange  [fmin fmax] to restrict the centre frequency (optional)

if nargin < 8
    offRange = [];
end
s = double(seg(:));
if any(m)
    pAct = mean(abs(s(m)).^2);
else
    pAct = mean(abs(s).^2);
end
s = s*sqrt(pOn/max(pAct, eps));

fadName = "none";
if useFading
    [s, fadName] = p2_applyFading(s, cfg.fs);
end

f0 = 0;
if useOffset
    maxOff = max(0, cfg.fs/2*cfg.bandEdgeUse - bw/2);
    if isempty(offRange)
        f0 = (2*rand - 1)*maxOff;
    else
        lo = max(offRange(1), -maxOff);
        hi = min(offRange(2),  maxOff);
        if hi < lo
            f0 = min(max(mean(offRange), -maxOff), maxOff);
        else
            f0 = lo + rand*(hi - lo);
        end
    end
    f0 = f0 + (2*rand - 1)*cfg.cfoMax;
    s = s.*exp(1j*2*pi*f0*(0:numel(s)-1)'/cfg.fs);
end
end
