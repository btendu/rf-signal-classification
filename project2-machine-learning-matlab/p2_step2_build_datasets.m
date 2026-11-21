%% STEP 2 - Experimental scenarios and feature extraction
% Handout: "The ML models can initially classify individual signals and
% subsequently be tested under coexistence scenarios ... The study can
% progressively introduce: 1. different SNR levels 2. frequency offsets
% 3. AWGN and multipath fading 4. adjacent-channel interference
% 5. co-channel interference 6. multiple simultaneously active signals."
%
% Builds, and extracts the 52 handcrafted features of, every frame of:
%   datasets/level1.mat ... level6.mat   7 classes x cfg.framesPerClass
%   datasets/sweep.mat                   fixed SNRs -10..20 dB (level 3)
%   datasets/coex1.mat ... coex5.mat     the five coexistence scenarios
%
% Each file holds X (features), y (labels), meta (table of the random
% settings of every frame), featureNames, featureGroups and a stratified
% train/test split (isTest).

cfg = p2Config();
S = load(fullfile(cfg.poolDir, "pool.mat"), "pool");
pool = S.pool;
if ~isfolder(cfg.dataDir), mkdir(cfg.dataDir); end
[~, featureNames, featureGroups] = p2_extractFeatures( ...
    complex(randn(cfg.frameLen,1), randn(cfg.frameLen,1)), cfg.fs, cfg.psdNfft, cfg.stftNfft);
fprintf("%d features per frame.\n", numel(featureNames));

%% Single-signal data sets, levels 1-6
for L = 1:6
    file = fullfile(cfg.dataDir, sprintf("level%d.mat", L));
    nPer = cfg.framesPerClass;
    if localExists(file, nPer*numel(cfg.classNames), cfg), continue; end
    labels = repelem(cfg.classNames(:), nPer, 1);
    n = numel(labels);
    X = zeros(n, numel(featureNames));
    metas = cell(n, 1);
    tStart = tic;
    parfor i = 1:n
        rng(cfg.seed + 1e6*L + i, "twister");
        [x, m] = p2_makeFrame(cfg, pool, labels(i), L);
        X(i, :) = p2_extractFeatures(x, cfg.fs, cfg.psdNfft, cfg.stftNfft);
        metas{i} = m;
    end
    meta = struct2table([metas{:}]');
    localSave(file, X, labels, meta, featureNames, featureGroups, cfg, L);
    fprintf("  %s: %d frames in %.1f min\n", cfg.levelNames(L), n, toc(tStart)/60);
end

%% SNR sweep (level-3 conditions, fixed SNRs)
file = fullfile(cfg.dataDir, "sweep.mat");
nPer = cfg.sweepFramesPerClass;
nS = numel(cfg.sweepSNR);
if ~localExists(file, nPer*numel(cfg.classNames)*nS, cfg)
    [cc, ss] = ndgrid(1:numel(cfg.classNames), 1:nS);
    cc = repelem(cc(:), nPer, 1);
    ss = repelem(ss(:), nPer, 1);
    labels = cfg.classNames(cc).';
    snrs = cfg.sweepSNR(ss).';
    n = numel(labels);
    X = zeros(n, numel(featureNames));
    metas = cell(n, 1);
    tStart = tic;
    parfor i = 1:n
        rng(cfg.seed + 7e6 + i, "twister");
        [x, m] = p2_makeFrame(cfg, pool, labels(i), cfg.sweepLevel, struct("snr", snrs(i)));
        X(i, :) = p2_extractFeatures(x, cfg.fs, cfg.psdNfft, cfg.stftNfft);
        m.sweepSNR = snrs(i);           % also set for noise frames
        metas{i} = m;
    end
    meta = struct2table([metas{:}]');
    localSave(file, X, labels, meta, featureNames, featureGroups, cfg, 0, true);
    fprintf("  SNR sweep: %d frames in %.1f min\n", n, toc(tStart)/60);
end

%% Coexistence scenarios
for k = 1:size(cfg.coexPairs, 1)
    A = cfg.coexPairs(k, 1);
    B = cfg.coexPairs(k, 2);
    file = fullfile(cfg.dataDir, sprintf("coex%d.mat", k));
    nPer = cfg.coexFramesPerLabel;
    if localExists(file, 4*nPer, cfg), continue; end
    kinds = repelem(["A" "B" "AB" "Noise"]', nPer, 1);
    n = numel(kinds);
    X = zeros(n, numel(featureNames));
    metas = cell(n, 1);
    tStart = tic;
    parfor i = 1:n
        rng(cfg.seed + 1e7*k + i, "twister");
        [x, m] = p2_makeCoexFrame(cfg, pool, A, B, kinds(i));
        X(i, :) = p2_extractFeatures(x, cfg.fs, cfg.psdNfft, cfg.stftNfft);
        metas{i} = m;
    end
    meta = struct2table([metas{:}]');
    labels = meta.label;
    localSave(file, X, labels, meta, featureNames, featureGroups, cfg, 100 + k);
    fprintf("  Coexistence %s vs %s: %d frames in %.1f min\n", A, B, n, toc(tStart)/60);
end

%% Feature-extraction time (one CPU core)
nT = 200;
xs = cell(nT, 1);
labels = cfg.classNames(randi(numel(cfg.classNames), nT, 1));
for i = 1:nT
    xs{i} = p2_makeFrame(cfg, pool, labels(i), 3);
end
tStart = tic;
for i = 1:nT
    p2_extractFeatures(xs{i}, cfg.fs, cfg.psdNfft, cfg.stftNfft);
end
featureTimeMs = toc(tStart)/nT*1e3;
save(fullfile(cfg.dataDir, "timing.mat"), "featureTimeMs");
fprintf("Feature extraction: %.2f ms per frame (frame length %.3f ms).\n", ...
    featureTimeMs, cfg.frameLen/cfg.fs*1e3);

%% Example frames: every class at level 1 and at level 6
for L = [1 6]
    fig = figure("Color","w", "Position",[50 50 1600 420]);
    tl = tiledlayout(fig, 1, numel(cfg.classNames), "TileSpacing","compact");
    title(tl, "Example frames, " + cfg.levelNames(L));
    for k = 1:numel(cfg.classNames)
        rng(cfg.seed + 99*k + L, "twister");
        [x, m] = p2_makeFrame(cfg, pool, cfg.classNames(k), L);
        ttl = cfg.classNames(k);
        if m.numInterf > 0
            ttl = ttl + " (+" + m.interfClasses + ")";
        end
        p2_showFrame(nexttile(tl), x, cfg.fs, ttl);
    end
    p2_saveFigure(fig, fullfile(cfg.resultsDir, sprintf("p2_step2_examples_L%d.png", L)), Resolution=150);
end

% -------------------------------------------------------------------------
function tf = localExists(file, n, cfg)
tf = false;
if isfile(file) && ~cfg.forceRegenerate
    S = load(file, "X");
    tf = size(S.X, 1) == n;
    if tf
        fprintf("  %s already exists - reused.\n", file);
    end
end
end

function localSave(file, X, labels, meta, featureNames, featureGroups, cfg, seedOffset, noSplit)
if nargin < 9, noSplit = false; end
y = categorical(string(labels));
rng(cfg.seed + seedOffset, "twister");
if noSplit
    isTest = true(numel(y), 1);          % sweep set is used for testing only
else
    cvp = cvpartition(y, "HoldOut", cfg.testFraction);
    isTest = test(cvp);
end
save(file, "X", "y", "meta", "featureNames", "featureGroups", "isTest");
end
