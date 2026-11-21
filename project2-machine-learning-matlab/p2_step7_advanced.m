%% STEP 7 - Advanced extensions
% Handout "Advanced Extensions":
%   - Hybrid ML ensemble: RF + XGBoost + SVM            -> step 4 (model "Hybrid")
%   - Automatic feature selection                        -> step 3
%   - Classification of overlapping spectra              -> step 5d
%   - Low-SNR signal classification                      -> 7c below
%   - Incremental learning / online learning             -> 7a below
%   - Transferability across frequency bands             -> 7b below
%   - Real-time inference implementation                 -> 7d below
%   - Comparison of shallow ML models (accuracy,
%     robustness, latency, computational cost)           -> step 5e
%
% Output: results/p2_step7_*.csv/png and results/p2_step7_results.mat

cfg = p2Config();
S = load(fullfile(cfg.poolDir, "pool.mat"), "pool");
pool = S.pool;
nF = numel(cfg.classNames);
nFeat = numel(load(fullfile(cfg.dataDir, "level1.mat"), "featureNames").featureNames);

%% 7a - incremental (online) learning
% A Naive Bayes model is first trained on level-1 frames only and then
% updated chunk by chunk with level-6 frames, as if the radio environment
% changed while the classifier is running. After every chunk it is tested
% on the level-6 test set. Compared with: no update, and full retraining.
D1 = load(fullfile(cfg.dataDir, "level1.mat"));
D6 = load(fullfile(cfg.dataDir, "level6.mat"));
[X1, st] = p2_standardize(D1.X(~D1.isTest, :));
X6tr = p2_standardize(D6.X(~D6.isTest, :), st);
X6te = p2_standardize(D6.X(D6.isTest, :), st);
y1 = D1.y(~D1.isTest);
y6tr = D6.y(~D6.isTest);
y6te = D6.y(D6.isTest);
cats = categories(D6.y);
rng(cfg.seed, "twister");
order = randperm(numel(y6tr));
X6tr = X6tr(order, :);  y6tr = y6tr(order);
chunk = cfg.incrementalChunk;
nChunks = floor(numel(y6tr)/chunk);
incAcc = nan(nChunks + 1, 1);
incMethod = "";
try
    base = fitcnb(X1 + 1e-6*randn(size(X1)), y1, 'ClassNames', cats);
    inc = incrementalLearner(base);
    incMethod = "incrementalClassificationNaiveBayes (incrementalLearner + updateMetricsAndFit)";
    incAcc(1) = mean(predict(inc, X6te) == y6te);
    tUpd = 0;
    for c = 1:nChunks
        idx = (c-1)*chunk + (1:chunk);
        t0 = tic;
        inc = updateMetricsAndFit(inc, X6tr(idx, :), y6tr(idx));
        tUpd = tUpd + toc(t0);
        incAcc(c + 1) = mean(predict(inc, X6te) == y6te);
    end
    updateMsPerChunk = 1e3*tUpd/nChunks;
catch err
    fprintf("  incrementalLearner not available (%s);\n  using a running-statistics Gaussian Naive Bayes instead.\n", err.message);
    incMethod = "own running-statistics Gaussian Naive Bayes (incrementalLearner not available)";
    nb = localNbInit(X1, y1, cats);
    incAcc(1) = mean(localNbPredict(nb, X6te, cats) == y6te);
    tUpd = 0;
    for c = 1:nChunks
        idx = (c-1)*chunk + (1:chunk);
        t0 = tic;
        nb = localNbUpdate(nb, X6tr(idx, :), y6tr(idx), cats);
        tUpd = tUpd + toc(t0);
        incAcc(c + 1) = mean(localNbPredict(nb, X6te, cats) == y6te);
    end
    updateMsPerChunk = 1e3*tUpd/nChunks;
end
fullNB = p2_trainModel("NaiveBayes", [X1; X6tr], [y1; y6tr], cfg);
accFullRetrain = mean(p2_predictModel(fullNB, X6te) == y6te);
fullRF = p2_trainModel("RandomForest", [X1; X6tr], [y1; y6tr], cfg);
accFullRetrainRF = mean(p2_predictModel(fullRF, X6te) == y6te);
fprintf("Incremental learning: L1-only %.3f -> after %d level-6 frames %.3f (full retrain NB %.3f, RF %.3f)\n", ...
    incAcc(1), nChunks*chunk, incAcc(end), accFullRetrain, accFullRetrainRF);
incTable = table((0:nChunks)'*chunk, incAcc, 'VariableNames', ["Level6FramesSeen" "Accuracy"]);
writetable(incTable, fullfile(cfg.resultsDir, "p2_step7a_incremental.csv"));
fig = figure("Color","w", "Position",[50 50 800 450]);
plot(incTable.Level6FramesSeen, 100*incAcc, "-o", "LineWidth", 1.3); hold on
yline(100*accFullRetrain, "--", "NB retrained on all data");
yline(100*accFullRetrainRF, ":", "RF retrained on all data");
grid on; ylim([0 100]);
xlabel("Level-6 frames streamed into the model"); ylabel("Accuracy on level-6 test set (%)");
title("Online learning: model trained on level 1 adapts to level 6");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step7a_incremental.png"), Resolution=150);

%% 7b - transferability across frequency bands
% Train on signals whose centre frequency is in the lower half of the band,
% test on signals in the upper half (and the other way round), compared
% with training and testing in the same half. Level-3 conditions.
lim = cfg.fs/2*cfg.bandEdgeUse;
bands = {[-lim 0], [0 lim]};
bandNames = ["lower half" "upper half"];
nPer = cfg.transferFramesPerClass;
Xb = cell(1, 2);  yb = cell(1, 2);
for b = 1:2
    labels = repelem(cfg.classNames(:), nPer, 1);
    Xt = zeros(numel(labels), nFeat);
    rngBase = cfg.seed + 3e7 + 1e5*b;
    offR = bands{b};
    parfor i = 1:numel(labels)
        rng(rngBase + i, "twister");
        x = p2_makeFrame(cfg, pool, labels(i), 3, struct("offsetRange", offR));
        Xt(i, :) = p2_extractFeatures(x, cfg.fs, cfg.psdNfft, cfg.stftNfft);
    end
    Xb{b} = Xt;  yb{b} = categorical(labels);
end
rng(cfg.seed, "twister");
trRows = {};
for name = ["RandomForest" "SVM" "GradientBoosting"]
    for b = 1:2
        cv = cvpartition(yb{b}, "HoldOut", 0.3);
        [Xtr, stb] = p2_standardize(Xb{b}(training(cv), :));
        mdl = p2_trainModel(name, Xtr, yb{b}(training(cv)), cfg);
        same = mean(p2_predictModel(mdl, p2_standardize(Xb{b}(test(cv), :), stb)) == yb{b}(test(cv)));
        o = 3 - b;
        other = mean(p2_predictModel(mdl, p2_standardize(Xb{o}, stb)) == yb{o});
        trRows(end+1, :) = {name, "train " + bandNames(b), same, other, same - other}; %#ok<SAGROW>
    end
end
transferTable = cell2table(trRows, 'VariableNames', ["Model" "Training" ...
    "AccSameBand" "AccOtherBand" "TransferLoss"]);
transferTable.Model = string(transferTable.Model);
disp(transferTable);
writetable(transferTable, fullfile(cfg.resultsDir, "p2_step7b_transfer_bands.csv"));
fig = figure("Color","w", "Position",[50 50 850 420]);
bar(categorical(transferTable.Model + " / " + transferTable.Training), ...
    100*[transferTable.AccSameBand transferTable.AccOtherBand]);
legend(["Tested in the same half-band" "Tested in the other half-band"], ...
    "Location","southoutside", "NumColumns", 2);
ylabel("Accuracy (%)"); ylim([0 100]); grid on
title("Transferability across frequency bands");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step7b_transfer_bands.png"), Resolution=150);

%% 7c - low-SNR specialist
% Train on frames with SNR -10..5 dB only and compare, on the SNR-sweep
% test set, with the general level-3 models (trained on -10..30 dB).
nPer = cfg.lowSnrFramesPerClass;
labels = repelem(cfg.classNames(:), nPer, 1);
XL = zeros(numel(labels), nFeat);
r = cfg.lowSnrRange;
parfor i = 1:numel(labels)
    rng(cfg.seed + 4e7 + i, "twister");
    snr = r(1) + rand*diff(r);
    x = p2_makeFrame(cfg, pool, labels(i), 3, struct("snr", snr));
    XL(i, :) = p2_extractFeatures(x, cfg.fs, cfg.psdNfft, cfg.stftNfft);
end
yL = categorical(labels);
[XLs, stL] = p2_standardize(XL);
low = struct();
for name = ["RandomForest" "GradientBoosting" "SVM"]
    low.(name) = p2_trainModel(name, XLs, yL, cfg);
end
low.Hybrid = p2_trainModel("Hybrid", [], [], cfg, low);
SW = load(fullfile(cfg.dataDir, "sweep.mat"));
M3 = load(fullfile(cfg.modelDir, "models_L3.mat"));
snrList = cfg.sweepSNR(cfg.sweepSNR <= 10);
lowRows = {};
for name = ["RandomForest" "Hybrid"]
    ypG = p2_predictModel(M3.models.(name), p2_standardize(SW.X, M3.st));
    ypL = p2_predictModel(low.(name), p2_standardize(SW.X, stL));
    for s = snrList
        sel = SW.meta.sweepSNR == s;
        lowRows(end+1, :) = {name, s, mean(ypG(sel) == SW.y(sel)), mean(ypL(sel) == SW.y(sel))}; %#ok<SAGROW>
    end
end
lowTable = cell2table(lowRows, 'VariableNames', ["Model" "SNR_dB" "GeneralModel" "LowSNRModel"]);
lowTable.Model = string(lowTable.Model);
disp(lowTable);
writetable(lowTable, fullfile(cfg.resultsDir, "p2_step7c_low_snr.csv"));
fig = figure("Color","w", "Position",[50 50 800 450]);
hold on
for name = ["RandomForest" "Hybrid"]
    t = lowTable(lowTable.Model == name, :);
    plot(t.SNR_dB, 100*t.GeneralModel, "--o", "DisplayName", name + " general (-10..30 dB)");
    plot(t.SNR_dB, 100*t.LowSNRModel, "-s", "LineWidth", 1.3, "DisplayName", name + " low-SNR specialist");
end
grid on; ylim([0 100]); legend("Location","southeast");
xlabel("In-band SNR (dB)"); ylabel("Accuracy (%)");
title("Low-SNR classification");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step7c_low_snr.png"), Resolution=150);

%% 7d - real-time inference
% One frame at a time, as a real receiver would: feature extraction plus
% one prediction, measured on one CPU core, compared with the frame length.
M6 = load(fullfile(cfg.modelDir, "models_L6.mat"));
nT = cfg.latencyFrames;
xs = cell(nT, 1);
labs = cfg.classNames(randi(nF, nT, 1));
for i = 1:nT
    xs{i} = p2_makeFrame(cfg, pool, labs(i), 6);
end
frameMs = cfg.frameLen/cfg.fs*1e3;
featMs = zeros(nT, 1);
F = zeros(nT, nFeat);
for i = 1:nT
    t0 = tic;
    F(i, :) = p2_extractFeatures(xs{i}, cfg.fs, cfg.psdNfft, cfg.stftNfft);
    featMs(i) = 1e3*toc(t0);
end
Fs = p2_standardize(F, M6.st);
rtRows = {};
for name = cfg.modelNames
    p2_predictModel(M6.models.(name), Fs(1, :));          % warm-up
    pm = zeros(nT, 1);
    for i = 1:nT
        t0 = tic;
        p2_predictModel(M6.models.(name), Fs(i, :));
        pm(i) = 1e3*toc(t0);
    end
    tot = featMs + pm;
    rtRows(end+1, :) = {name, median(featMs), median(pm), median(tot), ...
        prctile(tot, 95), frameMs/median(tot), median(tot) <= frameMs}; %#ok<SAGROW>
end
rtTable = cell2table(rtRows, 'VariableNames', ["Model" "Features_ms" "Predict_ms" ...
    "Total_ms_median" "Total_ms_p95" "FramesPerFrameTime" "RealTime"]);
rtTable.Model = string(rtTable.Model);
disp(rtTable);
fprintf("Frame duration: %.3f ms. Feature extraction: %.2f ms (median).\n", frameMs, median(featMs));
writetable(rtTable, fullfile(cfg.resultsDir, "p2_step7d_realtime.csv"));
fig = figure("Color","w", "Position",[50 50 900 450]);
bar(categorical(rtTable.Model, rtTable.Model), [rtTable.Features_ms rtTable.Predict_ms], "stacked");
yline(frameMs, "r--", sprintf("frame length %.3f ms", frameMs));
set(gca, "YScale","log"); grid on
legend(["Feature extraction" "Prediction"], "Location","northwest");
ylabel("Time per frame (ms, one CPU core, log scale)");
title("Single-frame (real-time) processing time");
p2_saveFigure(fig, fullfile(cfg.resultsDir, "p2_step7d_realtime.png"), Resolution=150);

save(fullfile(cfg.resultsDir, "p2_step7_results.mat"), "incTable", "incMethod", ...
    "accFullRetrain", "accFullRetrainRF", "updateMsPerChunk", "transferTable", ...
    "lowTable", "rtTable", "frameMs");

% =========================================================================
% Fallback online Gaussian Naive Bayes (running means and variances)
function nb = localNbInit(X, y, cats)
nb.n  = zeros(numel(cats), 1);
nb.mu = zeros(numel(cats), size(X, 2));
nb.M2 = zeros(numel(cats), size(X, 2));
nb = localNbUpdate(nb, X, y, cats);
end

function nb = localNbUpdate(nb, X, y, cats)
for k = 1:numel(cats)
    Xk = X(y == cats{k}, :);
    for i = 1:size(Xk, 1)                     % Welford update
        nb.n(k) = nb.n(k) + 1;
        d = Xk(i, :) - nb.mu(k, :);
        nb.mu(k, :) = nb.mu(k, :) + d/nb.n(k);
        nb.M2(k, :) = nb.M2(k, :) + d.*(Xk(i, :) - nb.mu(k, :));
    end
end
end

function yp = localNbPredict(nb, X, cats)
v = nb.M2./max(nb.n - 1, 1) + 1e-6;
prior = nb.n/sum(nb.n);
L = zeros(size(X, 1), numel(cats));
for k = 1:numel(cats)
    L(:, k) = log(prior(k)) - 0.5*sum(log(2*pi*v(k, :)) + (X - nb.mu(k, :)).^2./v(k, :), 2);
end
[~, i] = max(L, [], 2);
yp = categorical(cats(i), cats);
end
