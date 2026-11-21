%% STEP 3.1 - Create the data set for training and validation
% Handout: "Create a data set to be used for training and validation.
% Create signals with a variety of impairments."
%
% Every frame is 20 ms of the 2.4 GHz band, 60 MHz wide, containing
% Bluetooth, WLAN, both (in separate parts of the band) or noise only.
% Each frame gets a random draw of these impairments:
%
%   - noise:            SNR from 5 to 40 dB, chosen separately per signal
%   - multipath fading: TGn indoor models A-E for WLAN, Rician per hop for
%                       Bluetooth
%   - frequency error:  transmitter oscillator offset on each signal
%   - phase noise:      receiver oscillator
%   - IQ imbalance:     receiver amplitude and phase mismatch
%   - DC offset:        LO leakage spike in the middle of the band
%   - timing:           random start time, packet lengths, gaps and hops
%
% The frame is turned into a 256x256 spectrogram image, and a label image
% of the same size records which class every pixel belongs to. Because the
% signals are synthetic, the labels are exact; nothing is labelled by hand.
%
% Split: 80 % training, 10 % validation (used during training),
%        10 % test (kept aside for step 3.3).
%
% Time: this is the slowest step after training. With Parallel Computing
% Toolbox all cores are used. To try the chain quickly first, set
% cfg.numFrames = 100 in projectConfig.

cfg = projectConfig();
counts = round(cfg.numFrames*cfg.split/sum(cfg.split));
names  = ["train" "val" "test"];

fprintf("Generating %d frames into %s\n", sum(counts), cfg.dataDir);
for k = 1:numel(names)
    % A different seed block per split, so no frame appears twice.
    generateDatasetSplit(cfg, names(k), counts(k), ...
        cfg.seed + 100000*k, cfg.overlapFraction);
end

%% Show a few training frames with their labels
imds = imageDatastore(fullfile(cfg.dataDir,"train","images"), FileExtensions=".png");
pxds = pixelLabelDatastore(fullfile(cfg.dataDir,"train","labels"), ...
    cfg.classNames, cfg.labelIDs, FileExtensions=".png");
meta = readtable(fullfile(cfg.dataDir,"train","meta.csv"), TextType="string");

if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
for wanted = ["Bluetooth" "WLAN" "Both"]
    i = find(meta.type == wanted, 1);
    if isempty(i), continue; end
    fig = showSegmentation(readimage(imds,i), cfg, ...
        sprintf("Training frame %d (%s)", i, wanted), ...
        "Labels", readimage(pxds,i));
    saveFigure(fig, fullfile(cfg.resultsDir, ...
        "step3a_example_" + lower(wanted) + ".png"), Resolution=150);
end

%% How the pixels are shared between the classes
tbl = countEachLabel(pxds);
disp(tbl);
figure("Color","w", "Name","Step 3.1 - class balance");
bar(categorical(tbl.Name), 100*tbl.PixelCount/sum(tbl.PixelCount));
ylabel("Share of all training pixels (%)");
title("Class balance (noise dominates, so training uses class weights)");
saveFigure(gcf, fullfile(cfg.resultsDir,"step3a_class_balance.png"), Resolution=150);
