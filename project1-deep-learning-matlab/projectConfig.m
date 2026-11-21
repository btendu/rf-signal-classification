function cfg = projectConfig()
%PROJECTCONFIG  Every setting for Project 1 in one place.
%
%   cfg = projectConfig() returns a struct used by all the step scripts.
%   Change a value here and every step picks it up.
%
%   Signal pair chosen from the handout list: Bluetooth and WiFi (WLAN),
%   which share the 2.4 GHz ISM band and interfere with one another.

root = fileparts(mfilename("fullpath"));
addpath(fullfile(root,"helpers"));

% ---------------------------------------------------------------- Folders
cfg.rootDir       = root;
cfg.waveformDir   = fullfile(root,"waveforms");   % step 1 and 2 outputs
cfg.dataDir       = fullfile(root,"data");        % step 3.1 data set
cfg.pretrainedDir = fullfile(root,"pretrained");  % put the example's network here
cfg.modelDir      = fullfile(root,"model");       % step 3.2 trained network
cfg.resultsDir    = fullfile(root,"results");     % figures and metrics
cfg.captureDir    = fullfile(root,"captures");    % SDR recordings (step 4)

% ------------------------------------------- What the receiver looks at
% 60 MHz is used instead of the example's 61.44 MHz because it is an exact
% multiple of both native signal rates (20 MHz WLAN, 4 MHz Bluetooth), so
% the Multiband Combiner interpolates by whole numbers (3 and 15).
cfg.fs              = 60e6;      % sample rate of every frame (Hz)
cfg.frameDuration   = 20e-3;     % one frame = one spectrogram (s)
cfg.centerFrequency = 2.44e9;    % middle of the 2.4 GHz ISM band (Hz)
cfg.imageSize       = [256 256]; % [time rows, frequency columns]
cfg.dynamicRangeDb  = 50;        % colour scale span above the noise floor

% ------------------------------------------------------------ Classes
cfg.classNames = ["Noise" "Bluetooth" "WLAN"];
cfg.labelIDs   = [0 1 2];        % pixel values stored in the label PNGs

% ---------------------------------------------------- Bluetooth BR/EDR
cfg.btSamplesPerSymbol = 4;                         % 1 Msym/s -> 4 MHz
cfg.btRate      = cfg.btSamplesPerSymbol*1e6;       % native rate (Hz)
cfg.btBandwidth = 1e6;           % labelled width of one hop (Hz)
cfg.btMaxOffset = 28e6;          % hops lie within +/- this of centre (Hz)
cfg.btHopCount  = [2 6];         % hop frequencies used inside one frame
cfg.btActivity  = [0.25 0.9];    % fraction of slots that carry a packet
cfg.btMaxCFO    = 75e3;          % transmitter frequency error (Hz)

% --------------------------------------------------------------- WLAN
cfg.wlanRate       = 20e6;       % native rate of a 20 MHz channel (Hz)
cfg.wlanMaxOffset  = 20e6;       % channel centre within +/- this (Hz)
cfg.wlanMaxCFO     = 50e3;       % about 20 ppm at 2.4 GHz (Hz)
cfg.wlanFormatProb = [0.4 0.4 0.2];   % [802.11a/g OFDM, 802.11n HT, 802.11b DSSS]
cfg.wlanIdleRange  = [0.2e-3 6e-3];   % largest gap between packets (s)
cfg.wlanNominalBandwidth = 16.6e6;    % used only to define the SNR (Hz)

% Bluetooth hops stay at least this far from the WLAN centre when the two
% signals must not overlap in frequency (handout step 2).
cfg.nonOverlapGuard = 12e6;

% ------------------------------------------------------------ Data set
cfg.numFrames     = 1500;                  % total frames generated
cfg.split         = [0.8 0.1 0.1];         % train / validation / test
cfg.frameTypeProb = [0.25 0.25 0.45 0.05]; % Bluetooth, WLAN, Both, Noise
cfg.overlapFraction = 0;     % share of "Both" frames allowed to overlap
cfg.seed          = 2026;    % makes the data set repeatable
cfg.forceRegenerate = false; % true = rebuild even if the files exist

% --------------------------------------------------------- Impairments
cfg.snrRangeBT      = [5 40];    % dB, measured inside the 1 MHz hop
cfg.snrRangeWLAN    = [5 40];    % dB, measured inside the WLAN channel
cfg.pFading         = 0.7;       % chance of multipath fading
cfg.pPhaseNoise     = 0.25;      % chance of oscillator phase noise
cfg.pIQImbalance    = 0.7;       % chance of IQ imbalance
cfg.iqAmpRangeDb    = 0.2;       % +/- amplitude imbalance (dB)
cfg.iqPhaseRangeDeg = 1.5;       % +/- phase imbalance (degrees)
cfg.pDCOffset       = 0.7;       % chance of a DC (LO leakage) spike
cfg.dcRangeDb       = [-30 -8];  % DC power relative to total noise (dB)

% ------------------------------------------------------------ Training
cfg.baseNetwork        = "resnet18";  % same backbone as the example
cfg.pretrainedFile     = "specSenseTrainedNetResnet18.mat";
cfg.miniBatchSize      = 16;     % lower this if the GPU runs out of memory
cfg.maxEpochs          = 20;
cfg.initialLearnRate   = 0.01;
cfg.learnRateDropPeriod = 8;
cfg.headLearnRateBoost = 5;      % new output layers learn this much faster
cfg.networkFile        = fullfile(cfg.modelDir,"rfClassifierNet.mat");

% ------------------------------------------------- Over-the-air (step 4)
cfg.ota.radioID      = "usb:0";  % ADALM-PLUTO address
cfg.ota.numCaptures  = 5;        % frames to capture and classify
cfg.ota.transmit     = false;    % true = also radiate the step-2 waveform
cfg.ota.txGain       = -30;      % dB, keep low; see README before changing

% ------------------------------------------- Overrides set by RUN_ME.m
% RUN_ME chooses a smaller run when there is no GPU. It stores its choice
% here so that every step script sees the same settings.
ov = getappdata(0, "RFP1_overrides");
if isstruct(ov)
    names = fieldnames(ov);
    for i = 1:numel(names)
        cfg.(names{i}) = ov.(names{i});
    end
end

% ------------------------------------------------------- Sanity checks
assert(mod(cfg.fs,cfg.btRate) == 0 && mod(cfg.fs,cfg.wlanRate) == 0, ...
    "cfg.fs must be a whole multiple of the Bluetooth and WLAN rates.");
assert(cfg.btMaxOffset + cfg.btRate/2 <= cfg.fs/2, ...
    "Bluetooth hops would fall outside the captured band.");
assert(cfg.wlanMaxOffset + cfg.wlanRate/2 <= cfg.fs/2, ...
    "The WLAN channel would fall outside the captured band.");
end
