function cfg = p2Config()
%P2CONFIG  Every setting for Project 2 in one place.
%
%   Project 2: Machine Learning-Based RF Signal Classification for
%   Spectrum Sensing and Interference Identification.
%
%   cfg = p2Config() returns a struct used by all the p2_step scripts.
%   Change a value here and every step picks it up.

root = fileparts(mfilename("fullpath"));
addpath(fullfile(root,"p2helpers"));

% ---------------------------------------------------------------- Folders
cfg.rootDir    = root;
cfg.poolDir    = fullfile(root,"pool");       % generated base waveforms
cfg.dataDir    = fullfile(root,"datasets");   % feature tables
cfg.modelDir   = fullfile(root,"models");     % trained models
cfg.resultsDir = fullfile(root,"results");    % figures and tables
cfg.captureDir = fullfile(root,"captures");   % SDR recordings

% ------------------------------------------------ What the receiver sees
% 30.72 MHz is the native rate of 20 MHz LTE and of 5G NR with a 2048- or
% 1024-point FFT, and wide enough for a 20 MHz WiFi channel.
cfg.fs       = 30.72e6;   % sample rate of every frame (Hz)
cfg.frameLen = 8192;      % samples per frame = 0.267 ms

% ------------------------------------------------------------ Classes
% Noise = no signal present (needed to measure false alarms and missed
% detections). The six signal classes are the ones named in the handout.
cfg.classNames    = ["Noise" "WLAN" "Bluetooth" "ZigBee" "LTE" "NR5G" "Radar"];
cfg.signalClasses = cfg.classNames(2:end);

% --------------------------------------------------- Base waveform pool
cfg.poolSize          = 16;     % different random waveforms per class
cfg.poolDuration      = 4e-3;   % length of each waveform (s)
cfg.minActiveFraction = 0.3;    % a frame must contain this much signal

% ------------------------------------------ Impairment levels (handout)
% The handout's "the study can progressively introduce" list. Each level
% keeps everything from the levels before it.
cfg.levelNames = ["L1 Different SNR levels" ...
                  "L2 + Frequency offsets" ...
                  "L3 + AWGN and multipath fading" ...
                  "L4 + Adjacent-channel interference" ...
                  "L5 + Co-channel interference" ...
                  "L6 + Multiple simultaneously active signals"];
cfg.snrRange       = [-10 30];  % in-band SNR of the wanted signal (dB)
cfg.cfoMax         = 20e3;      % oscillator error added to every offset (Hz)
cfg.bandEdgeUse    = 0.95;      % signals stay inside 95 % of +/- fs/2
cfg.sirRange       = [0 20];    % wanted / interferer power (dB)
cfg.aciGuard       = 0.5e6;     % gap between adjacent signals (Hz)
cfg.maxInterferers = 3;         % level 6: 1 to 3 interferers
cfg.framesPerClass = 400;       % frames per class per level
cfg.testFraction   = 0.25;      % held out for testing

% ----------------------------------------------- SNR sweep (robustness)
cfg.sweepSNR            = -10:2:20;
cfg.sweepFramesPerClass = 100;
cfg.sweepLevel          = 3;    % realistic channel, no interference

% ------------------------------------------ Coexistence (handout list)
cfg.coexPairs = ["Bluetooth" "WLAN";
                 "ZigBee"    "Bluetooth";
                 "ZigBee"    "WLAN";
                 "Radar"     "NR5G";
                 "LTE"       "NR5G"];
cfg.coexFramesPerLabel = 300;
cfg.coexSnrRange       = [0 30];
cfg.coexSirRange       = [-10 10];

% ------------------------------------------------------------ Features
cfg.psdNfft  = 512;   % Welch PSD resolution: 60 kHz
cfg.stftNfft = 128;   % spectrogram resolution: 240 kHz, 4.2 us per slice

% --------------------------------------------------- Feature selection
cfg.selectionLevel     = 6;
cfg.selectionFolds     = 5;
cfg.selectionK         = [1 2 3 4 5 6 8 10 12 15 20 25 30 40];
cfg.selectionTolerance = 0.01;  % "as good as the best" = within 1 point
cfg.selectionTrees     = 60;
cfg.corrThreshold      = 0.90;  % correlation filter cut-off

% -------------------------------------------------------------- Models
cfg.modelNames = ["SVM" "RandomForest" "KNN" "DecisionTree" "NaiveBayes" ...
                  "LogisticRegression" "GradientBoosting" "AdaBoost" "Hybrid"];
cfg.numTrees    = 150;   % random forest
cfg.numBoost    = 150;   % boosting rounds
cfg.boostSplits = 20;    % size of each boosted tree
cfg.knnK        = 5;

% ----------------------------------------------- SDR validation (step 6)
% mode: "labeled"    transmit each class through a cable and classify it
%       "ambient"    receive only and report what is on air
%       "recordings" classify the MAT-files in the captures folder
cfg.sdr.mode            = "labeled";
cfg.sdr.rxRadioID       = "usb:0";
cfg.sdr.txRadioID       = "usb:0";   % same PLUTO can transmit and receive
cfg.sdr.centerFrequency = 2.44e9;
cfg.sdr.txGain          = -30;       % dB; see README before changing
cfg.sdr.framesPerClass  = 50;
cfg.sdr.ambientFrames   = 200;
cfg.sdr.modelLevel      = 6;         % which level's models to test

% ---------------------------------------------------- Advanced (step 7)
cfg.transferFramesPerClass = 200;   % per half-band, for the transfer test
cfg.lowSnrFramesPerClass   = 200;   % training set for the low-SNR model
cfg.lowSnrRange            = [-10 5];
cfg.latencyFrames          = 100;
cfg.incrementalChunk       = 50;

% -------------------------------------------------------------- Others
cfg.seed            = 2026;
cfg.forceRegenerate = false;   % true = rebuild files that already exist

% ------------------------------------------ Overrides set by RUN_ME.m
ov = getappdata(0, "RFP2_overrides");
if isstruct(ov)
    names = fieldnames(ov);
    for i = 1:numel(names)
        cfg.(names{i}) = ov.(names{i});
    end
end

assert(cfg.frameLen >= 4*cfg.psdNfft, "Frame too short for the PSD size.");
end
