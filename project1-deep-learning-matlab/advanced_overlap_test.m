%% ADVANCED (optional) - Signals that overlap in frequency
% Handout, advanced possibilities: "Train and test with signals that
% overlap in frequency."
%
% Part 1 (this script as it stands): TEST. Generate frames in which the
% Bluetooth hops are allowed to land on top of the WLAN channel, and
% measure how the network trained on non-overlapping signals copes.
%
% Part 2: TRAIN. To train with overlapping signals as well, set
%     cfg.overlapFraction = 0.5;  cfg.forceRegenerate = true;
% in projectConfig, then rerun step3a, step3b and this script, and compare
% the two sets of numbers.

cfg = projectConfig();
S   = load(cfg.networkFile);
net = S.net;

%% Overlap test set: every frame has both signals, overlap allowed
numFrames = 150;
cfgOverlap = cfg;
cfgOverlap.frameTypeProb = [0 0 1 0];          % "Both" only
generateDatasetSplit(cfgOverlap, "overlap_test", numFrames, cfg.seed + 900000, 1);

testDir = fullfile(cfg.dataDir, "overlap_test");
imds = imageDatastore(fullfile(testDir,"images"), FileExtensions=".png");
pxdsTruth = pixelLabelDatastore(fullfile(testDir,"labels"), ...
    cfg.classNames, cfg.labelIDs, FileExtensions=".png");

%% Evaluate
predDir = fullfile(cfg.resultsDir, "overlap_predictions");
if ~isfolder(predDir), mkdir(predDir); end
pxdsPred = semanticseg(imds, net, Classes=cfg.classNames, ...
    MiniBatchSize=cfg.miniBatchSize, WriteLocation=predDir);
metricsOverlap = evaluateSemanticSegmentation(pxdsPred, pxdsTruth, ...
    Metrics=["global-accuracy" "accuracy" "iou"]);

fprintf("\nOverlapping signals:\n");
disp(metricsOverlap.DataSetMetrics);
disp(metricsOverlap.ClassMetrics);

baseFile = fullfile(cfg.resultsDir, "step3c_metrics.mat");
if isfile(baseFile)
    B = load(baseFile);
    fprintf("For comparison, non-overlapping test set (step 3.3):\n");
    disp(B.metrics.DataSetMetrics);
    disp(B.metrics.ClassMetrics);
end
save(fullfile(cfg.resultsDir,"advanced_overlap_metrics.mat"), "metricsOverlap");

%% One example where a hop really does land on the WLAN channel
meta = readtable(fullfile(testDir,"meta.csv"), TextType="string");
i = find(meta.numBT > 0, 1);
if ~isempty(i)
    fig = showSegmentation(readimage(imds,i), cfg, ...
        sprintf("Overlap test frame %d", i), ...
        "True labels", readimage(pxdsTruth,i), ...
        "Predicted labels", readimage(pxdsPred,i));
    saveFigure(fig, fullfile(cfg.resultsDir,"advanced_overlap_example.png"), ...
        Resolution=150);
end
