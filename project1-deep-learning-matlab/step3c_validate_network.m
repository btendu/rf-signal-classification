%% STEP 3.3 - Validate the network with a small part of the generated data
% Handout: "Validate your AI network with a small percentage of your
% generated data."
%
% 10 % of the generated frames (the "test" folder) were never shown to the
% network, neither for training nor for choosing when to stop. This script
% measures how well the trained network labels them.
%
% Outputs in results/:  step3c_metrics.mat, a confusion matrix, example
% frames and a plot of accuracy against SNR.

cfg = projectConfig();
S   = load(cfg.networkFile);
net = S.net;

testDir = fullfile(cfg.dataDir, "test");
imds = imageDatastore(fullfile(testDir,"images"), FileExtensions=".png");
pxdsTruth = pixelLabelDatastore(fullfile(testDir,"labels"), ...
    cfg.classNames, cfg.labelIDs, FileExtensions=".png");

%% Label every test frame with the network
if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
predDir = fullfile(cfg.resultsDir, "test_predictions");
if ~isfolder(predDir), mkdir(predDir); end
pxdsPred = semanticseg(imds, net, Classes=cfg.classNames, ...
    MiniBatchSize=cfg.miniBatchSize, WriteLocation=predDir, Verbose=true);

%% Metrics
metrics = evaluateSemanticSegmentation(pxdsPred, pxdsTruth, ...
    Metrics=["global-accuracy" "accuracy" "iou"]);
disp(metrics.DataSetMetrics);
disp(metrics.ClassMetrics);
save(fullfile(cfg.resultsDir,"step3c_metrics.mat"), "metrics");

figure("Color","w", "Name","Step 3.3 - confusion matrix");
cm = confusionchart(metrics.ConfusionMatrix.Variables, cfg.classNames, ...
    Normalization="row-normalized");
cm.Title = "Confusion matrix on held-out synthetic frames (per pixel)";
saveFigure(gcf, fullfile(cfg.resultsDir,"step3c_confusion.png"), Resolution=150);

%% Example frames: truth next to prediction
metaFile = fullfile(testDir, "meta.csv");
hasMeta  = isfile(metaFile);
if hasMeta
    meta = readtable(metaFile, TextType="string");
    show = [find(meta.type == "Bluetooth",1); find(meta.type == "WLAN",1); ...
            find(meta.type == "Both",1);      find(meta.type == "Both",1,"last")];
else
    show = (1:min(4,numel(imds.Files))).';
end
for i = unique(show(:)).'
    fig = showSegmentation(readimage(imds,i), cfg, ...
        sprintf("Test frame %d", i), ...
        "True labels", readimage(pxdsTruth,i), ...
        "Predicted labels", readimage(pxdsPred,i));
    saveFigure(fig, fullfile(cfg.resultsDir, ...
        sprintf("step3c_example_%03d.png", i)), Resolution=150);
end

%% Accuracy against SNR
% One point per test frame. The SNR used is the lower of the two signal
% SNRs in the frame, since the weaker signal is the harder one to find.
if hasMeta && height(meta) == height(metrics.ImageMetrics)
    snr = min([meta.snrBT meta.snrWLAN], [], 2, "omitnan");
    ok  = ~isnan(snr);
    edges   = 5:5:40;
    centers = edges(1:end-1) + 2.5;
    bin     = discretize(snr(ok), edges);
    iou     = metrics.ImageMetrics.MeanIoU(ok);
    keep    = ~isnan(bin) & ~isnan(iou);
    meanIoU = accumarray(bin(keep), iou(keep), [numel(centers) 1], @mean, NaN);

    figure("Color","w", "Name","Step 3.3 - accuracy against SNR");
    plot(snr(ok), iou, ".", "Color",[0.7 0.7 0.7]); hold on
    plot(centers, meanIoU, "-o", "LineWidth",1.5);
    xlabel("SNR of the weakest signal in the frame (dB)");
    ylabel("Mean IoU of the frame");
    legend("Each test frame","Average per 5 dB", "Location","southeast");
    grid on; title("Segmentation quality against SNR");
    saveFigure(gcf, fullfile(cfg.resultsDir,"step3c_iou_vs_snr.png"), Resolution=150);
end
