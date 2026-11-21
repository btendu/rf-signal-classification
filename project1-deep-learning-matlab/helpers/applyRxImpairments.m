function [y, imp] = applyRxImpairments(y, fs, cfg)
%APPLYRXIMPAIRMENTS  Receiver impairments and thermal noise for one frame.
%
%   [y, imp] = applyRxImpairments(y, fs, cfg)
%
%   Applied to the sum of all signals, in the order a real SDR front end
%   adds them: oscillator phase noise, IQ imbalance, DC offset (LO leakage)
%   and finally white noise of total power 1.

imp = struct("phaseNoise",false, "iqAmpDb",0, "iqPhaseDeg",0, "dcDb",NaN);
y = double(y(:));

if rand < cfg.pPhaseNoise
    level = [-80 -100 -120] - 10*rand;          % dBc/Hz
    pn = comm.PhaseNoise(Level=level, FrequencyOffset=[1e5 1e6 1e7], ...
        SampleRate=fs);
    y = pn(y);
    imp.phaseNoise = true;
end

if rand < cfg.pIQImbalance
    imp.iqAmpDb    = (2*rand-1)*cfg.iqAmpRangeDb;
    imp.iqPhaseDeg = (2*rand-1)*cfg.iqPhaseRangeDeg;
    y = iqimbal(y, imp.iqAmpDb, imp.iqPhaseDeg);
end

if rand < cfg.pDCOffset
    imp.dcDb = cfg.dcRangeDb(1) + rand*diff(cfg.dcRangeDb);
    y = y + 10^(imp.dcDb/20)*exp(1j*2*pi*rand);
end

y = y + sqrt(1/2)*(randn(size(y)) + 1j*randn(size(y)));
y = single(y);
end
