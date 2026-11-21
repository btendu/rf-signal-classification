function [seg, m, bw, subtype] = p2_drawSegment(entries, N, cfg)
%P2_DRAWSEGMENT  A random N-sample window from a random pool waveform.
%
%   [seg, m, bw, subtype] = p2_drawSegment(entries, N, cfg)
%
%   The window is chosen so that the signal is actually on for a useful
%   part of it (cfg.minActiveFraction, or less for low-duty-cycle signals
%   such as radar, so that at least one pulse is included).

e = entries{randi(numel(entries))};
L = numel(e.x);
mask = logical(e.mask(:));
thresh = min(cfg.minActiveFraction, 0.8*mean(mask));
cm = cumsum([0; double(mask)]);
best = 1;
bestFrac = -1;
for t = 1:50
    s0 = randi(L - N + 1);
    frac = (cm(s0+N) - cm(s0))/N;
    if frac > bestFrac
        best = s0;
        bestFrac = frac;
    end
    if frac >= thresh
        break
    end
end
seg = double(e.x(best:best+N-1));
m   = mask(best:best+N-1);
bw  = e.bw;
subtype = string(e.subtype);
end
