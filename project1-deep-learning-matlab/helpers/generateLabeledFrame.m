function [img, lbl, y, meta] = generateLabeledFrame(cfg, frameType, allowOverlap)
%GENERATELABELEDFRAME  One impaired wideband frame plus its pixel labels.
%
%   [img, lbl, y, meta] = generateLabeledFrame(cfg, frameType, allowOverlap)
%
%   frameType     "Bluetooth", "WLAN", "Both" or "Noise"
%   allowOverlap  true lets Bluetooth hop on top of the WLAN channel
%                 (advanced work); false keeps them apart in frequency.
%
%   img   256x256x3 uint8 spectrogram image (network input)
%   lbl   256x256 uint8 label image, values from cfg.labelIDs
%   y     complex single samples at cfg.fs (what an SDR would deliver)
%   meta  the random settings used, for later analysis
%
%   Signal chain for each signal:
%     waveform -> multipath fading -> frequency error -> Multiband Combiner
%     (frequency translation to its place in the band) -> power scaling
%   then for the sum: phase noise, IQ imbalance, DC offset and noise.

if nargin < 3
    allowOverlap = false;
end
frameType = string(frameType);
fs = cfg.fs;
T  = cfg.frameDuration;
N  = round(T*fs);

hasWLAN = any(frameType == ["WLAN" "Both"]);
hasBT   = any(frameType == ["Bluetooth" "Both"]);
idBT    = cfg.labelIDs(cfg.classNames == "Bluetooth");
idWLAN  = cfg.labelIDs(cfg.classNames == "WLAN");

y     = complex(zeros(N,1,"single"));
boxes = zeros(0,5);              % [tStart tEnd fLow fHigh labelID]
meta  = struct("type",frameType, "overlap",false, ...
    "snrWLAN",NaN, "snrBT",NaN, "fcWLAN",NaN, "numWLAN",0, "numBT",0, ...
    "fading",false, "cfoWLAN",NaN, "cfoBT",NaN, ...
    "phaseNoise",false, "iqAmpDb",0, "iqPhaseDeg",0, "dcDb",NaN);
fcW = NaN;

% Noise power over the full 60 MHz is fixed at 1, so a signal with in-band
% SNR "snr" and bandwidth B needs on-air power 10^(snr/10) * B/fs.

% --------------------------------------------------------------- WLAN
if hasWLAN
    [xw, spans] = genWlanFrame(cfg);

    if rand < cfg.pFading        % TGn indoor multipath models A-E
        if hasWlanToolbox()
            profiles = ["Model-A" "Model-B" "Model-C" "Model-D" "Model-E"];
            chan = wlanTGnChannel(SampleRate=cfg.wlanRate, ...
                DelayProfile=profiles(randi(numel(profiles))), ...
                CarrierFrequency=cfg.centerFrequency, ...
                NumTransmitAntennas=1, NumReceiveAntennas=1, ...
                LargeScaleFadingEffect="None");
            xw = single(chan(double(xw)));
        else
            xw = single(tgnLikeChannel(xw, cfg.wlanRate));
        end
        meta.fading = true;
    end

    fcW  = (2*rand-1)*cfg.wlanMaxOffset;         % place in the band
    cfoW = (2*rand-1)*cfg.wlanMaxCFO;            % transmitter error
    yw   = localTranslate(xw, cfg.wlanRate, fcW + cfoW, fs, N);

    duty = sum(spans(:,2) - spans(:,1) + 1)/size(xw,1);
    snrW = cfg.snrRangeWLAN(1) + rand*diff(cfg.snrRangeWLAN);
    pOn  = 10^(snrW/10)*cfg.wlanNominalBandwidth/fs;
    yw   = yw*sqrt(pOn/(mean(abs(yw).^2)/duty + realmin("single")));
    y    = y + yw;

    nW = size(spans,1);
    boxes = [boxes; (spans(:,1)-1)/cfg.wlanRate, spans(:,2)/cfg.wlanRate, ...
        fcW - spans(:,3)/2, fcW + spans(:,3)/2, repmat(idWLAN,nW,1)];
    meta.snrWLAN = snrW;
    meta.fcWLAN  = fcW;
    meta.cfoWLAN = cfoW;
    meta.numWLAN = nW;
end

% ---------------------------------------------------------- Bluetooth
if hasBT
    candidates = -cfg.btMaxOffset:1e6:cfg.btMaxOffset;   % 1 MHz channels
    overlapHere = hasWLAN && allowOverlap;
    if hasWLAN && ~allowOverlap
        candidates = candidates(abs(candidates - fcW) > cfg.nonOverlapGuard);
    end
    K   = min(numel(candidates), randi(cfg.btHopCount));
    hop = candidates(randperm(numel(candidates), K));
    activity = cfg.btActivity(1) + rand*diff(cfg.btActivity);

    [xb, bb] = genBluetoothFrame(cfg, hop, activity);
    if ~isempty(bb)
        cfoB = (2*rand-1)*cfg.btMaxCFO;
        yb   = localTranslate(xb, cfg.btRate, hop + cfoB, fs, N);

        duty = sum(bb(:,2) - bb(:,1))/T;
        snrB = cfg.snrRangeBT(1) + rand*diff(cfg.snrRangeBT);
        pOn  = 10^(snrB/10)*cfg.btBandwidth/fs;
        yb   = yb*sqrt(pOn/(mean(abs(yb).^2)/duty + realmin("single")));
        y    = y + yb;

        % Bluetooth rows go last so that, where the signals overlap, the
        % narrow Bluetooth burst is labelled on top of the WLAN packet.
        boxes = [boxes; bb, repmat(idBT,size(bb,1),1)];
        meta.snrBT   = snrB;
        meta.cfoBT   = cfoB;
        meta.numBT   = size(bb,1);
        meta.overlap = overlapHere;
    end
end

% ----------------------------------- Receiver impairments and noise
[y, imp] = applyRxImpairments(y, fs, cfg);
meta.phaseNoise = imp.phaseNoise;
meta.iqAmpDb    = imp.iqAmpDb;
meta.iqPhaseDeg = imp.iqPhaseDeg;
meta.dcDb       = imp.dcDb;

% ------------------------------------------------ Image and labels
img = rfSpectrogramImage(y, cfg.imageSize, cfg.dynamicRangeDb);
lbl = boxesToLabelImage(boxes, fs, T, cfg.imageSize);
end

% -------------------------------------------------------------------------
function y = localTranslate(x, fsIn, offsets, fsOut, N)
% Frequency-translate each column of x to its offset and add them, using
% the Multiband Combiner named in the handout. The combiner interpolates
% from fsIn to fsOut first, so the shifted signals do not alias.
mbc = comm.MultibandCombiner(InputSampleRate=fsIn, ...
    FrequencyOffsets=offsets(:).', ...
    OutputSampleRateSource="Property", OutputSampleRate=fsOut);
y = mbc(x);
release(mbc);
y = y(:);
if numel(y) >= N
    y = y(1:N);
else
    y(end+1:N,1) = 0;
end
end
