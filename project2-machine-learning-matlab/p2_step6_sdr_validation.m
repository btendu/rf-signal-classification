%% STEP 6 - Validation with real signals from an SDR (ADALM-PLUTO)
% Handout: "the trained ML models can be validated using real-world RF
% signals captured from SDR platforms such as ADALM-PLUTO or USRP. Such
% validation can help assess the gap between simulated datasets and real
% wireless environments."
%
% cfg.sdr.mode (p2Config.m):
%   "labeled"     The PLUTO transmits each class in turn through a cable
%                 (TX -> 30 dB attenuator -> RX) and receives it again.
%                 Every captured frame has a known label, so the accuracy on
%                 real hardware can be compared with the simulated accuracy
%                 = the sim-to-real gap.
%   "ambient"     Receive only (antenna on RX) at cfg.sdr.centerFrequency
%                 and report what the models detect on air (no labels).
%   "recordings"  Classify the MAT-files in the "captures" folder. Each file
%                 holds x (complex column, sampled at 30.72 MHz) and
%                 optionally label (one of cfg.classNames).
%
% Every capture is saved in "captures", so the step can be repeated later
% without the radio (mode "recordings").
% With no radio and no recordings the step writes a note and ends:
% nothing is simulated or made up here.

cfg = p2Config();
if ~isfolder(cfg.captureDir), mkdir(cfg.captureDir); end
M = load(fullfile(cfg.modelDir, sprintf("models_L%d.mat", cfg.sdr.modelLevel)));
R4 = load(fullfile(cfg.resultsDir, "p2_step4_results.mat"));
noteFile = fullfile(cfg.resultsDir, "p2_step6_sdr_note.txt");
mode = string(cfg.sdr.mode);

hasPluto = exist("sdrrx", "file") == 2 || exist("sdrrx", "file") == 6;
radioOK = false;
if hasPluto && mode ~= "recordings"
    try
        radios = findPlutoRadio();
        radioOK = ~isempty(radios);
    catch
        radioOK = false;
    end
end
recFiles = dir(fullfile(cfg.captureDir, "*.mat"));
if ~radioOK
    if isempty(recFiles)
        msg = "No ADALM-PLUTO radio was found (or the PLUTO support package is not installed) " + ...
            "and the captures folder is empty, so SDR validation was skipped. " + ...
            "See README section 6 to run it with a radio or with recordings.";
        fprintf("%s\n", msg);
        writelines(msg, noteFile);
        return
    end
    if mode ~= "recordings"
        fprintf("No radio found - classifying the %d recordings in the captures folder.\n", numel(recFiles));
    end
    mode = "recordings";
end

N  = cfg.frameLen;
fs = cfg.fs;
captures = struct("x", {}, "label", {}, "source", {});

%% Acquire
switch mode
    case "labeled"
        S = load(fullfile(cfg.poolDir, "pool.mat"), "pool");
        rx = localRx(cfg, N);
        tx = sdrtx("Pluto", "RadioID", cfg.sdr.txRadioID, ...
            "CenterFrequency", cfg.sdr.centerFrequency, ...
            "BasebandSampleRate", fs, "Gain", cfg.sdr.txGain);
        for cls = cfg.classNames
            if cls == "Noise"
                release(tx);                              % transmitter off
            else
                w = localTxWaveform(S.pool.(cls), cfg);
                transmitRepeat(tx, w);
            end
            pause(0.5);
            for i = 1:5, rx(); end                        % let the AGC settle
            n = 0;
            while n < cfg.sdr.framesPerClass
                [x, len, overrun] = rx();
                if len < N || overrun, continue; end
                x = double(x);
                % keep only frames that contain the signal
                if cls ~= "Noise" && localActive(x) < 0.2, continue; end
                n = n + 1;
                captures(end+1) = struct("x", x, "label", cls, "source", "PLUTO labeled"); %#ok<SAGROW>
            end
            fprintf("  captured %d frames of %s\n", n, cls);
        end
        release(tx); release(rx);
    case "ambient"
        rx = localRx(cfg, N);
        for i = 1:5, rx(); end
        while numel(captures) < cfg.sdr.ambientFrames
            [x, len] = rx();
            if len < N, continue; end
            captures(end+1) = struct("x", double(x), "label", "", "source", "PLUTO ambient"); %#ok<SAGROW>
        end
        release(rx);
    otherwise   % recordings
        for f = recFiles'
            Sx = load(fullfile(f.folder, f.name));
            if ~isfield(Sx, "x"), continue; end
            lab = "";
            if isfield(Sx, "label"), lab = string(Sx.label); end
            xr = double(Sx.x(:));
            for s = 1:N:numel(xr) - N + 1          % cut long recordings into frames
                captures(end+1) = struct("x", xr(s:s+N-1), "label", lab, ...
                    "source", "file " + f.name); %#ok<SAGROW>
            end
        end
end
if mode ~= "recordings"
    stamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    save(fullfile(cfg.captureDir, "capture_" + mode + "_" + stamp + ".mat"), "captures", "fs");
end
if isempty(captures)
    writelines("SDR validation: no usable frames were captured.", noteFile);
    return
end

%% Classify
X = zeros(numel(captures), numel(M.featureNames));
for i = 1:numel(captures)
    X(i, :) = p2_extractFeatures(captures(i).x, fs, cfg.psdNfft, cfg.stftNfft);
end
Xs = p2_standardize(X, M.st);
labels = string({captures.label})';
hasLabels = all(labels ~= "");
rows = {};
predAll = struct();
for name = cfg.modelNames
    yp = p2_predictModel(M.models.(name), Xs);
    predAll.(name) = yp;
    if hasLabels
        yt = categorical(labels, M.cats);
        m = p2_metrics(yt, yp, M.cats);
        simAcc = R4.results.Accuracy(R4.results.Model == name & R4.results.Level == cfg.sdr.modelLevel);
        rows(end+1, :) = {name, simAcc, m.accuracy, simAcc - m.accuracy, m.f1}; %#ok<SAGROW>
    end
end

if hasLabels
    gap = cell2table(rows, 'VariableNames', ["Model" "SimulatedAccuracy" ...
        "RealAccuracy" "SimToRealGap" "RealMacroF1"]);
    gap.Model = string(gap.Model);
    disp(gap);
    writetable(gap, fullfile(cfg.resultsDir, "p2_step6_sim_to_real_gap.csv"));
    [~, b] = max(gap.RealAccuracy);
    best = gap.Model(b);
    fig = figure("Color","w", "Position",[50 50 650 560]);
    cm = confusionchart(categorical(labels, M.cats), predAll.(best), "Normalization","row-normalized");
    cm.Title = "Real captures (" + mode + "), " + best;
    p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step6_confusion_real.png"), Resolution=150);
    fig = figure("Color","w", "Position",[50 50 900 420]);
    bar(categorical(gap.Model, gap.Model), 100*[gap.SimulatedAccuracy gap.RealAccuracy]);
    legend(["Simulated test set" "Real SDR captures"], "Location","southoutside", "NumColumns", 2);
    ylabel("Accuracy (%)"); ylim([0 100]); grid on
    title("Sim-to-real gap");
    p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step6_sim_to_real_gap.png"), Resolution=150);
else
    best = R4.bestModel;
    yp = predAll.(best);
    counts = countcats(yp);
    det = table(string(M.cats(:)), counts(:), 100*counts(:)/numel(yp), ...
        'VariableNames', ["Class" "Frames" "Percent"]);
    disp(det);
    writetable(det, fullfile(cfg.resultsDir, "p2_step6_detections_unlabeled.csv"));
    fig = figure("Color","w", "Position",[50 50 1400 420]);
    tl = tiledlayout(fig, 1, 4, "TileSpacing","compact");
    title(tl, "Received frames and " + best + " decisions");
    pick = round(linspace(1, numel(captures), 4));
    for i = pick
        p2_showFrame(nexttile(tl), captures(i).x, fs, string(yp(i)));
    end
    p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step6_examples_real.png"), Resolution=150);
end
writelines(sprintf("SDR validation ran in mode %s with %d frames.", mode, numel(captures)), noteFile);

% -------------------------------------------------------------------------
function rx = localRx(cfg, N)
rx = sdrrx("Pluto", "RadioID", cfg.sdr.rxRadioID, ...
    "CenterFrequency", cfg.sdr.centerFrequency, ...
    "BasebandSampleRate", cfg.fs, "SamplesPerFrame", N, ...
    "OutputDataType", "double", "GainSource", "AGC Slow Attack");
end

function w = localTxWaveform(entries, cfg)
% Concatenate several pool waveforms of one class into a transmit buffer.
w = [];
for k = 1:min(4, numel(entries))
    w = [w; double(entries{k}.x(:))]; %#ok<AGROW>
end
w = w/max(abs(w))*0.8;
w = w(1:min(end, 2^20));
end

function a = localActive(x)
% Fraction of the frame whose short-time power is well above the minimum.
p = movmean(abs(x).^2, 64);
a = mean(p > 4*prctile(p, 10));
end
