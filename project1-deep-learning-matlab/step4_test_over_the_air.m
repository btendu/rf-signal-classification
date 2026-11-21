%% STEP 3.4 - Test with over-the-air signals captured with an SDR
% Handout: "Test using over-the-air signals, captured with an SDR. You can
% also use an SDR to transmit the waveform, or capture existing signals
% from nearby cell towers, WiFi routers, etc."
%
% This script picks the first source that is available:
%
%   1. An ADALM-PLUTO radio. It captures 60 MHz around 2.44 GHz, which
%      holds the WiFi routers and Bluetooth devices already on air nearby.
%      Set cfg.ota.transmit = true in projectConfig to also radiate the
%      step-2 waveform from the same PLUTO (read the README note first).
%   2. Recordings in the "captures" folder (MAT-files with variables iq
%      and fs), for example made earlier or with a USRP.
%   3. If neither exists, one simulated frame, clearly marked as such, so
%      that the processing chain can still be tried. That is NOT an
%      over-the-air result.
%
% Each capture is turned into a spectrogram with the same function used
% for training, labelled by the network, displayed and summarised.

cfg = projectConfig();
S   = load(cfg.networkFile);
net = S.net;

N = round(cfg.frameDuration*cfg.fs);          % samples per capture
outDir = fullfile(cfg.resultsDir, "ota");
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(cfg.captureDir), mkdir(cfg.captureDir); end

%% Find a signal source
havePluto = false;
if exist("findPlutoRadio","file") == 2
    havePluto = ~isempty(findPlutoRadio());
end
captureFiles = dir(fullfile(cfg.captureDir, "*.mat"));

captures = {};        % each entry: struct with iq, fs, source
if havePluto
    tx = [];
    if cfg.ota.transmit
        W = load(fullfile(cfg.waveformDir,"combinedWaveform.mat"));
        txWave = 0.8*W.combined/max(abs(W.combined));    % stay inside [-1,1]
        tx = sdrtx("Pluto", RadioID=char(cfg.ota.radioID), ...
            CenterFrequency=cfg.centerFrequency, ...
            BasebandSampleRate=cfg.fs, Gain=cfg.ota.txGain);
        transmitRepeat(tx, txWave);
        fprintf("Transmitting the step-2 waveform at %.0f dB gain.\n", cfg.ota.txGain);
    end

    rx = sdrrx("Pluto", RadioID=char(cfg.ota.radioID), ...
        CenterFrequency=cfg.centerFrequency, ...
        BasebandSampleRate=cfg.fs, ...
        SamplesPerFrame=N, ...
        OutputDataType="single", ...
        GainSource="AGC Slow Attack", ...
        EnableBurstMode=true, NumFramesInBurst=1);
    for k = 1:3                                 % let the gain control settle
        rx();
    end
    for k = 1:cfg.ota.numCaptures
        [iq, valid, overflow] = rx();
        if ~valid
            warning("Capture %d was not valid and is skipped.", k);
            continue
        end
        if overflow
            warning("Capture %d lost samples (overflow).", k);
        end
        fs = cfg.fs;  fc = cfg.centerFrequency;
        stamp = char(datetime("now", Format="yyyyMMdd_HHmmss"));
        file  = fullfile(cfg.captureDir, sprintf("pluto_%s_%02d.mat", stamp, k));
        save(file, "iq", "fs", "fc");
        captures{end+1} = struct("iq",iq, "fs",fs, ...
            "source",sprintf("PLUTO capture %d",k)); %#ok<SAGROW>
        pause(0.2);
    end
    release(rx);
    if ~isempty(tx), release(tx); end

elseif ~isempty(captureFiles)
    for k = 1:numel(captureFiles)
        C = load(fullfile(captureFiles(k).folder, captureFiles(k).name));
        if ~isfield(C,"iq") || ~isfield(C,"fs")
            warning("%s has no iq/fs variables and is skipped.", captureFiles(k).name);
            continue
        end
        captures{end+1} = struct("iq",C.iq(:), "fs",double(C.fs), ...
            "source","Recording " + string(captureFiles(k).name)); %#ok<SAGROW>
    end

else
    warning("rfproject:noSDR", ...
        "No PLUTO radio and no recordings found. Showing ONE SIMULATED frame " + ...
        "so the chain can be tried. This is not an over-the-air test.");
    rng("shuffle");
    [~, ~, iq] = generateLabeledFrame(cfg, "Both", false);
    captures{1} = struct("iq",iq, "fs",cfg.fs, "source","SIMULATED frame (not over the air)");
end

%% Classify every capture
for k = 1:numel(captures)
    iq = double(captures{k}.iq(:));
    if captures{k}.fs < cfg.fs
        % A narrower recording would leave part of the image empty, which
        % the network never saw in training. See README, "Other radios".
        warning("%s was sampled at %.2f MHz, below the %.0f MHz the network " + ...
            "was trained for, and is skipped.", captures{k}.source, ...
            captures{k}.fs/1e6, cfg.fs/1e6);
        continue
    elseif captures{k}.fs > cfg.fs
        % For example 61.44 MHz: resample to 60 MHz (keeps the middle 60 MHz).
        [p, q] = rat(cfg.fs/captures{k}.fs);
        iq = resample(iq, p, q);
    end
    if numel(iq) < N
        warning("%s is shorter than one frame and is skipped.", captures{k}.source);
        continue
    end
    iq = iq(1:N);

    img = rfSpectrogramImage(iq, cfg.imageSize, cfg.dynamicRangeDb);
    [predicted, confidence] = semanticseg(img, net, Classes=cfg.classNames);

    fprintf("\n%s\n", captures{k}.source);
    summary = summarizeSegmentation(predicted, cfg);
    fprintf("  Mean confidence of the labels: %.2f\n", mean(confidence(:)));

    fig = showSegmentation(img, cfg, captures{k}.source, ...
        "Network labels", predicted);
    saveFigure(fig, fullfile(outDir, sprintf("ota_%02d.png",k)), Resolution=150);
    source = captures{k}.source;
    save(fullfile(outDir, sprintf("ota_%02d.mat",k)), "predicted", "summary", "source");
end
