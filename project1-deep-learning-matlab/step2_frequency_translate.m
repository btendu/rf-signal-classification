%% STEP 2 - Frequency-translate the signals so they do not overlap
% Handout: "Frequency translate one or both signals so that they do not
% overlap in frequency. Use MATLAB's Multiband Combiner for this."
%
% Both waveforms from step 1 sit at 0 Hz (baseband), so added together they
% would lie on top of each other. comm.MultibandCombiner
%   1. interpolates both from 20 MHz to 60 MHz, so there is room to move;
%   2. shifts each one by its own frequency offset;
%   3. adds them into one wideband signal.
%
% Output: waveforms/combinedWaveform.mat  (combined, fs, offsets)

cfg = projectConfig();
[bt,   fsBT]   = loadAppWaveform(fullfile(cfg.waveformDir,"bluetoothWaveform.mat"), 20e6);
[wlan, fsWLAN] = loadAppWaveform(fullfile(cfg.waveformDir,"wlanWaveform.mat"),      20e6);

%% Bring both to one input sample rate (the combiner needs equal rates)
fsIn = cfg.wlanRate;                                    % 20 MHz
if fsBT ~= fsIn
    [p, q] = rat(fsIn/fsBT);
    bt = resample(bt, p, q);
end
if fsWLAN ~= fsIn
    [p, q] = rat(fsIn/fsWLAN);
    wlan = resample(wlan, p, q);
end

%% Fit both into one 20 ms frame (repeat or cut as needed) at equal power
numIn = round(cfg.frameDuration*fsIn);
bt    = localFit(bt,   numIn);
wlan  = localFit(wlan, numIn);
bt    = bt/sqrt(mean(abs(bt(abs(bt) > 0)).^2));
wlan  = wlan/sqrt(mean(abs(wlan(abs(wlan) > 0)).^2));

%% Frequency translation with the Multiband Combiner
% Bluetooth moves to -15 MHz  -> occupies about -15.5 ... -14.5 MHz
% WLAN      moves to +12 MHz  -> occupies about  +3.1 ... +20.9 MHz
% Gap between them: about 17 MHz, so they do not overlap.
offsets = [-15e6 12e6];
mbc = comm.MultibandCombiner(InputSampleRate=fsIn, ...
    FrequencyOffsets=offsets, ...
    OutputSampleRateSource="Property", OutputSampleRate=cfg.fs);
combined = mbc([bt, wlan]);
release(mbc);

% For reference: the rate the combiner would pick on its own.
mbcAuto = comm.MultibandCombiner(InputSampleRate=fsIn, ...
    FrequencyOffsets=offsets, OutputSampleRateSource="Auto");
autoInfo = info(mbcAuto);
fprintf("Combiner output rate: %.0f MHz (automatic choice would be %.0f MHz)\n", ...
    cfg.fs/1e6, autoInfo.OutputSampleRate/1e6);

btBand   = offsets(1) + [-0.5 0.5]*1e6;
wlanBand = offsets(2) + [-8.9 8.9]*1e6;
fprintf("Bluetooth occupies %+.1f to %+.1f MHz\n", btBand/1e6);
fprintf("WLAN      occupies %+.1f to %+.1f MHz\n", wlanBand/1e6);
fprintf("Separation: %.1f MHz -> no overlap\n", (wlanBand(1) - btBand(2))/1e6);

fs = cfg.fs;
save(fullfile(cfg.waveformDir,"combinedWaveform.mat"), "combined", "fs", "offsets");

%% Show the result
figure("Color","w", "Name","Step 2 - before and after frequency translation");
tiledlayout(1,2, "TileSpacing","compact");

nexttile; hold on
[pB, f] = pwelch(bt,   [], [], 4096, fsIn, "centered");
pW      = pwelch(wlan, [], [], 4096, fsIn, "centered");
plot(f/1e6, 10*log10(pB)); plot(f/1e6, 10*log10(pW));
title("Before: both at baseband (they overlap)");
xlabel("Frequency (MHz)"); ylabel("Power density (dB/Hz)");
legend("Bluetooth","WLAN", "Location","south"); grid on

nexttile;
[pC, f] = pwelch(combined, [], [], 8192, cfg.fs, "centered");
plot(f/1e6, 10*log10(pC));
title("After: Multiband Combiner output");
xlabel("Frequency (MHz)"); grid on
xline(btBand/1e6, ":"); xline(wlanBand/1e6, ":");

if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
saveFigure(gcf, fullfile(cfg.resultsDir,"step2_spectrum.png"), Resolution=150);

% The same signal as the network will see it (a little noise is added only
% so that the colour scale has a noise floor to refer to).
noisy = combined + 0.01*sqrt(1/2)*(randn(size(combined)) + 1j*randn(size(combined)));
img = rfSpectrogramImage(noisy, cfg.imageSize, cfg.dynamicRangeDb);
C = cfg.imageSize(2);
figure("Color","w", "Name","Step 2 - spectrogram of the combined signal");
image((cfg.centerFrequency + ((0:C-1) - C/2)*cfg.fs/C)/1e6, ...
    [0 cfg.frameDuration*1e3], img);
xlabel("Frequency (MHz)"); ylabel("Time (ms)");
title("Combined signal: Bluetooth (left, narrow) and WLAN (right, wide)");
saveFigure(gcf, fullfile(cfg.resultsDir,"step2_spectrogram.png"), Resolution=150);

% -------------------------------------------------------------------------
function x = localFit(x, n)
% Repeat or cut the column vector x to exactly n samples.
x = x(:);
x = repmat(x, ceil(n/numel(x)), 1);
x = x(1:n);
end
