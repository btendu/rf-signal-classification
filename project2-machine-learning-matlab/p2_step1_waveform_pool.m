%% STEP 1 - Generate the RF waveforms in MATLAB
% Handout: "RF waveforms will be generated in MATLAB" for 5G, LTE, Wi-Fi,
% Bluetooth and ZigBee (plus radar for the "Radar vs. 5G" scenario).
%
% For every class, cfg.poolSize different random waveforms of 4 ms are
% generated and stored. Later steps cut random 0.27 ms frames out of them,
% so thousands of different frames can be made quickly.
%
%   WLAN       802.11a/g/n/b packets (WLAN Toolbox, or the project's own
%              802.11 generator when WLAN Toolbox is not available)
%   Bluetooth  BR/EDR packets (Bluetooth Toolbox)
%   ZigBee     IEEE 802.15.4 O-QPSK packets (own generator, written from
%              the standard)
%   LTE        E-UTRA test models, 5-20 MHz (LTE Toolbox, or LTE-numerology
%              OFDM when it is not available)
%   NR5G       5G NR downlink, 15/30 kHz SCS, 5-20 MHz (5G Toolbox, or
%              NR-numerology OFDM when it is not available)
%   Radar      pulsed LFM chirps and Barker-13 coded pulses (own generator)
%
% Output: pool/pool.mat and results/p2_step1_*.png

cfg = p2Config();
clear pool poolInfo
poolFile = fullfile(cfg.poolDir, "pool.mat");
if ~isfolder(cfg.poolDir), mkdir(cfg.poolDir); end
if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end

if isfile(poolFile) && ~cfg.forceRegenerate
    S = load(poolFile, "pool", "poolInfo");
    if isfield(S, "poolInfo") && S.poolInfo.poolSize == cfg.poolSize
        pool = S.pool;
        poolInfo = S.poolInfo;
        fprintf("Waveform pool already exists (%d per class) - reused.\n", cfg.poolSize);
    end
end

if ~exist("pool", "var")
    rng(cfg.seed, "twister");
    pool = struct();
    rows = {};
    for cls = cfg.signalClasses
        tStart = tic;
        entries = cell(cfg.poolSize, 1);
        for i = 1:cfg.poolSize
            entries{i} = p2_genPoolEntry(cls, cfg);
            rows(end+1, :) = {cls, entries{i}.subtype, entries{i}.bw/1e6, ...
                mean(entries{i}.mask), entries{i}.source}; %#ok<SAGROW>
        end
        pool.(cls) = entries;
        fprintf("  %-10s %d waveforms in %.1f s (%s)\n", cls, cfg.poolSize, ...
            toc(tStart), entries{1}.source);
    end
    poolInfo.poolSize = cfg.poolSize;
    poolInfo.table = cell2table(rows, 'VariableNames', ...
        ["Class" "Subtype" "BandwidthMHz" "DutyCycle" "Source"]);
    save(poolFile, "pool", "poolInfo", "-v7.3");
    writetable(poolInfo.table, fullfile(cfg.resultsDir, "p2_step1_pool_contents.csv"));
end

%% Summary per class
T = poolInfo.table;
T.Class = string(T.Class);
T.Source = string(T.Source);
summaryRows = {};
for cls = cfg.signalClasses
    r = T(T.Class == cls, :);
    summaryRows(end+1, :) = {cls, min(r.BandwidthMHz), max(r.BandwidthMHz), ...
        mean(r.DutyCycle), r.Source(1)}; %#ok<SAGROW>
end
poolSummary = cell2table(summaryRows, 'VariableNames', ...
    ["Class" "MinBandwidthMHz" "MaxBandwidthMHz" "MeanDutyCycle" "Source"]);
disp(poolSummary);
writetable(poolSummary, fullfile(cfg.resultsDir, "p2_step1_pool_summary.csv"));

%% One example of each class: spectrogram and spectrum
fig = figure("Color","w", "Position",[50 50 1500 700]);
tl = tiledlayout(fig, 2, numel(cfg.signalClasses), "TileSpacing","compact");
title(tl, "Step 1: one generated waveform of each class (first 1 ms, at band centre)");
nShow = round(1e-3*cfg.fs);
for k = 1:numel(cfg.signalClasses)
    cls = cfg.signalClasses(k);
    e = pool.(cls){1};
    x = double(e.x(1:nShow)) + 1e-3*(randn(nShow,1) + 1j*randn(nShow,1));
    p2_showFrame(nexttile(tl, k), x, cfg.fs, cls);
    ax = nexttile(tl, k + numel(cfg.signalClasses));
    [P, f] = pwelch(double(e.x), hann(1024), 512, 1024, cfg.fs, "centered");
    plot(ax, f/1e6, 10*log10(P/max(P)));
    ylim(ax, [-60 3]); grid(ax, "on");
    xlabel(ax, "Frequency (MHz)");
    if k == 1, ylabel(ax, "Relative PSD (dB)"); end
end
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step1_waveforms.png"), Resolution=150);
