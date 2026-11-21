%% STEP 3.2 - Train the network with transfer learning
% Handout: "Train your AI network on that generated data set. Use the
% existing network in the example and perform transfer learning."
%
% The network is the example's DeepLab v3+ / ResNet-18 semantic
% segmentation network. buildTransferNetwork keeps everything the example
% network learned and replaces only the output layers, which must change
% because our classes are Noise / Bluetooth / WLAN instead of the
% example's. Training then fine-tunes the whole network on our data.
%
% Before running: put specSenseTrainedNetResnet18.mat in the "pretrained"
% folder (README, "Getting the example network").
%
% Output: model/rfClassifierNet.mat

cfg = projectConfig();

%% Data
[cdsTrain, pxdsTrain, numTrain] = localDatastore(cfg, "train");
[cdsVal,   ~,         ~       ] = localDatastore(cfg, "val");

%% Network
[net0, report] = buildTransferNetwork(cfg);
fprintf("Starting point: %s\n", report.source);
fprintf("  %d of %d parameter arrays copied from the pretrained network.\n", ...
    report.numCopied, report.numParameters);
if ~isempty(report.newLayers)
    fprintf("  Layers trained from scratch: %s\n", strjoin(report.newLayers, ", "));
end
% To look at the network, or to edit it as in the handout's link "Transfer
% Learning with Deep Network Designer", run:   deepNetworkDesigner(net0)

%% Class weights (same method as the example)
% Noise covers most pixels and a Bluetooth burst only a few dozen, so each
% class is weighted by median frequency / its own frequency.
tbl = countEachLabel(pxdsTrain);
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
imageFreq(isnan(imageFreq)) = [];
classWeights = median(imageFreq) ./ imageFreq;
classWeights = classWeights/(sum(classWeights) + eps(class(classWeights)));
numClasses = numel(cfg.classNames);
if numel(classWeights) < numClasses
    classWeights = [classWeights; zeros(numClasses - numel(classWeights),1)];
end
disp(table(tbl.Name, classWeights, 'VariableNames', ["Class" "Weight"]));

%% Training options
% The learning rate is half the example's because most of the network is
% already trained and only needs adjusting.
opts = trainingOptions("sgdm", ...
    MiniBatchSize       = cfg.miniBatchSize, ...
    MaxEpochs           = cfg.maxEpochs, ...
    InitialLearnRate    = cfg.initialLearnRate, ...
    LearnRateSchedule   = "piecewise", ...
    LearnRateDropPeriod = cfg.learnRateDropPeriod, ...
    LearnRateDropFactor = 0.1, ...
    ValidationData      = cdsVal, ...
    ValidationFrequency = max(1, floor(numTrain/cfg.miniBatchSize)), ...
    ValidationPatience  = 5, ...
    Shuffle             = "every-epoch", ...
    OutputNetwork       = "best-validation-loss", ...
    Plots               = "training-progress", ...
    Verbose             = true);

%% Train
lossFcn = @(ypred,ytrue) weightedCrossEntropyLoss(ypred, ytrue, classWeights);
[net, trainInfo] = trainnet(cdsTrain, net0, lossFcn, opts);

%% Save
if ~isfolder(cfg.modelDir), mkdir(cfg.modelDir); end
classNames = cfg.classNames;
save(cfg.networkFile, "net", "classNames", "classWeights", "report", "trainInfo");
fprintf("Trained network saved to %s\n", cfg.networkFile);

% -------------------------------------------------------------------------
function [cds, pxds, n] = localDatastore(cfg, splitName)
imgDir = fullfile(cfg.dataDir, splitName, "images");
lblDir = fullfile(cfg.dataDir, splitName, "labels");
if ~isfolder(imgDir)
    error("rfproject:noData", ...
        "%s not found. Run step3a_create_dataset first.", imgDir);
end
imds = imageDatastore(imgDir, FileExtensions=".png");
pxds = pixelLabelDatastore(lblDir, cfg.classNames, cfg.labelIDs, ...
    FileExtensions=".png");
cds  = combine(imds, pxds);
n    = numel(imds.Files);
end
