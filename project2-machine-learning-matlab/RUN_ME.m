%% RUN_ME - runs the whole of Project 2 with one click
%
%   Project 2: Machine Learning-Based RF Signal Classification for
%   Spectrum Sensing and Interference Identification
%
%   Open this file in MATLAB and press Run (or type RUN_ME in the Command
%   Window while this folder is the Current Folder).
%
%   It will:
%     1. check your MATLAB release and toolboxes
%     2. run every step of the handout:
%          step 1  generate 5G, LTE, WiFi, Bluetooth, ZigBee and radar signals
%          step 2  six impairment levels + coexistence scenarios, 52 features
%          step 3  feature selection (correlation, MI, ANOVA, RFE, PCA)
%          step 4  train and compare SVM, RF, KNN, DT, NB, LR, GB, AdaBoost, Hybrid
%          step 5  robustness: SNR, bandwidth, interference, coexistence
%          step 6  SDR validation (only if an ADALM-PLUTO or recordings exist)
%          step 7  advanced: online learning, band transfer, low SNR, real time
%     3. save every figure and table in the "results" folder, a log of
%        everything printed, RESULTS_SUMMARY.txt with the numbers for the
%        report, and Project2_results_to_send.zip containing all of it
%
%   MODE:
%     "quick" - about 15-30 minutes, only to check that everything works
%     "full"  - the real run for the report (a few hours on a CPU)

MODE = "full";

% Which step to start from (1 = everything). RUN_FINISH.m uses this to
% continue after a stop without repeating the finished steps.
START_STEP = 1;

if exist("MODE_OVERRIDE", "var"),  MODE = MODE_OVERRIDE;        end
if exist("START_OVERRIDE", "var"), START_STEP = START_OVERRIDE; end
clear MODE_OVERRIDE START_OVERRIDE

%% ------------------------------------------------------------- setup
clc;
close all;
oldVisible = get(groot, "DefaultFigureVisible");
set(groot, "DefaultFigureVisible", "off");    % figures are saved, not shown
here = fileparts(mfilename("fullpath"));
cd(here);
addpath(here, fullfile(here, "p2helpers"));
if ~isfolder(fullfile(here, "results")), mkdir(fullfile(here, "results")); end
logFile = fullfile(here, "results", "command_window_log.txt");
if START_STEP == 1 && isfile(logFile), delete(logFile); end
diary(logFile);
tAll = tic;

fprintf("==============================================================\n");
fprintf(" Project 2: ML-based RF signal classification\n");
fprintf(" Started %s\n", string(datetime("now")));
fprintf("==============================================================\n\n");

%% ------------------------------------------------ 1. check the system
fprintf("[1/5] Checking MATLAB and toolboxes...\n");
if isMATLABReleaseOlderThan("R2022a")
    error("This project needs MATLAB R2022a or newer. You have %s.", version("-release"));
end
v = ver;
installed = string({v.Name});
required = ["Statistics and Machine Learning Toolbox" "Signal Processing Toolbox" ...
            "Communications Toolbox" "Bluetooth Toolbox"];
missing = required(~ismember(required, installed));
if ~isempty(missing)
    set(groot, "DefaultFigureVisible", oldVisible);
    diary("off");
    error("These toolboxes are missing. Install them (Home > Add-Ons) and run again:\n  %s", ...
        strjoin(missing, newline + "  "));
end
fprintf("      MATLAB %s, all required toolboxes found.\n", version("-release"));
optional = ["WLAN Toolbox" "LTE Toolbox" "5G Toolbox" "Parallel Computing Toolbox"];
for o = optional
    if ismember(o, installed)
        fprintf("      %-28s found\n", o);
    else
        fprintf("      %-28s not found - the project's own generator / serial loops are used\n", o);
    end
end
fprintf("\n");

%% ------------------------------------------------ 2. choose run size
fprintf("[2/5] Run size: %s\n\n", upper(MODE));
if MODE == "quick"
    ov = struct("poolSize",4, "framesPerClass",60, "sweepFramesPerClass",20, ...
        "coexFramesPerLabel",60, "numTrees",50, "numBoost",50, ...
        "selectionK",[1 2 3 5 8 12 20 30], "selectionFolds",3, "selectionTrees",30, ...
        "transferFramesPerClass",40, "lowSnrFramesPerClass",40, ...
        "latencyFrames",30, "incrementalChunk",20);
else
    ov = struct();
end
setappdata(0, "RFP2_overrides", ov);
cfg = p2Config();

%% ------------------------------------------------ 3. run the steps
steps = {
    "p2_step1_waveform_pool",         "Step 1: generate the RF signals",                 true
    "p2_step2_build_datasets",        "Step 2: scenarios, impairments and features",     true
    "p2_step3_feature_selection",     "Step 3: feature selection",                       true
    "p2_step4_train_compare_models",  "Step 4: train and compare the ML models",          true
    "p2_step5_robustness",            "Step 5: robustness and coexistence",               false
    "p2_step6_sdr_validation",        "Step 6: SDR validation (if hardware/recordings)", false
    "p2_step7_advanced",              "Step 7: advanced extensions",                     false
    };
status = strings(size(steps, 1), 1);
for k = 1:size(steps, 1)
    name = steps{k, 1};
    if k < START_STEP
        status(k) = "skipped (already done)";
        continue
    end
    fprintf("[3/5] (%d/%d) %s\n", k, size(steps, 1), steps{k, 2});
    fprintf("--------------------------------------------------------------\n");
    tStep = tic;
    try
        localRunStep(fullfile(here, name + ".m"));
        status(k) = sprintf("OK (%.1f min)", toc(tStep)/60);
    catch err
        status(k) = "FAILED: " + string(err.message);
        fprintf(2, "\n*** %s failed: %s\n", name, err.message);
        for s = 1:min(4, numel(err.stack))
            fprintf(2, "    in %s (line %d)\n", err.stack(s).name, err.stack(s).line);
        end
        if steps{k, 3}
            fprintf(2, "*** The next steps need this one, so the run stops here.\n");
            fprintf(2, "*** Send Project2_results_to_send.zip to get it fixed.\n\n");
            localWriteSummary(cfg, steps, status, MODE, toc(tAll));
            localPack(cfg);
            set(groot, "DefaultFigureVisible", oldVisible);
            diary("off");
            return
        end
    end
    close all force
    fprintf("\n");
end

%% ------------------------------------- 4-5. summary and packaging
fprintf("[4/5] Writing results/RESULTS_SUMMARY.txt...\n");
localWriteSummary(cfg, steps, status, MODE, toc(tAll));
fprintf("[5/5] Packing results...\n");
for k = 1:size(steps, 1)
    fprintf("      %-32s %s\n", steps{k, 1}, status(k));
end
fprintf("Finished in %.1f minutes.\n\n", toc(tAll)/60);
zipFile = localPack(cfg);
fprintf("==============================================================\n");
fprintf(" All figures and tables: %s\n", cfg.resultsDir);
fprintf(" Numbers for the report: %s\n", fullfile(cfg.resultsDir, "RESULTS_SUMMARY.txt"));
fprintf(" Send this file:         %s\n", zipFile);
fprintf("==============================================================\n");
set(groot, "DefaultFigureVisible", oldVisible);
diary("off");


%% =================================================================
%% local functions
%% =================================================================
function localRunStep(file)
% Run a step script in its own workspace so that variables do not clash.
run(file);
end

% -------------------------------------------------------------------------
function localWriteSummary(cfg, steps, status, mode, seconds)
f = fullfile(cfg.resultsDir, "RESULTS_SUMMARY.txt");
fid = fopen(f, "w");
closer = onCleanup(@() fclose(fid));
p = @(varargin) fprintf(fid, varargin{:});
showT = @(t) localShow(fid, t);

p("PROJECT 2 - RESULTS SUMMARY\n");
p("Generated %s, MATLAB %s, run size %s, total time %.1f min\n", ...
    string(datetime("now")), version("-release"), upper(mode), seconds/60);
p("Frames per class per level %d, pool %d waveforms per class, %d trees, %d boosting rounds\n\n", ...
    cfg.framesPerClass, cfg.poolSize, cfg.numTrees, cfg.numBoost);
p("STEP STATUS\n");
for k = 1:size(steps, 1)
    p("  %-32s %s\n", steps{k, 1}, status(k));
end
p("\n");

localSection(p, showT, fullfile(cfg.resultsDir, "p2_step1_pool_summary.csv"), "STEP 1 WAVEFORM POOL");
t = fullfile(cfg.dataDir, "timing.mat");
if isfile(t)
    T = load(t);
    p("Feature extraction time: %.3f ms per frame (frame = %.3f ms)\n\n", ...
        T.featureTimeMs, cfg.frameLen/cfg.fs*1e3);
end
s = fullfile(cfg.modelDir, "selection.mat");
if isfile(s)
    S = load(s);
    c = S.selection.chosen;
    p("STEP 3 FEATURE SELECTION\n");
    p("  Best CV accuracy %.4f; minimum subset %d features (%s), accuracy %.4f\n", ...
        S.selection.best, c.k, c.method, c.acc);
    p("  Subset: %s\n", strjoin(S.selection.featureNames(c.idx), ", "));
    p("  Correlation filter keeps %d features; PCA needs %d components for 95 %% variance\n\n", ...
        S.selection.corrKept, S.selection.nPC95);
    localSection(p, showT, fullfile(cfg.resultsDir, "p2_step3_accuracy_vs_features.csv"), "Accuracy vs number of features");
    localSection(p, showT, fullfile(cfg.resultsDir, "p2_step3_top_features.csv"), "Top features per ranking");
end
r = fullfile(cfg.resultsDir, "p2_step4_results.mat");
if isfile(r)
    R = load(r);
    p("STEP 4 MODELS - accuracy by level\n"); showT(R.accTable);
    p("Best model at level 6: %s\n\n", R.bestModel);
    p("All metrics, all levels\n"); showT(R.results);
    p("All features vs minimum subset (level 6)\n"); showT(R.subsetTable);
end
r = fullfile(cfg.resultsDir, "p2_step5_results.mat");
if isfile(r)
    R = load(r);
    p("STEP 5a ACCURACY VS SNR (best per-class model %s)\n", R.bestSnrModel); showT(R.snrTable);
    p("STEP 5b ACCURACY VS BANDWIDTH\n"); showT(R.bwTable);
    p("STEP 5c ACCURACY VS SIR\n"); showT(R.sirTable);
    p("Accuracy vs number of interferers\n"); showT(R.numTable);
    p("Accuracy vs interference type\n"); showT(R.typeTable);
    p("STEP 5d COEXISTENCE (all models)\n"); showT(R.coexTable);
    p("Coexistence, best model: identification of both signals vs separation and SIR\n"); showT(R.coexDetailTable);
    p("STEP 5e FINAL COMPARISON\n"); showT(R.comparison);
end
n = fullfile(cfg.resultsDir, "p2_step6_sdr_note.txt");
if isfile(n)
    p("STEP 6 SDR VALIDATION\n  %s\n", strjoin(readlines(n), " "));
    localSection(p, showT, fullfile(cfg.resultsDir, "p2_step6_sim_to_real_gap.csv"), "Sim-to-real gap");
    localSection(p, showT, fullfile(cfg.resultsDir, "p2_step6_detections_unlabeled.csv"), "Detections (unlabelled)");
    p("\n");
end
r = fullfile(cfg.resultsDir, "p2_step7_results.mat");
if isfile(r)
    R = load(r);
    p("STEP 7a INCREMENTAL LEARNING (%s)\n", R.incMethod);
    p("  Start %.4f, end %.4f, full retrain NB %.4f, RF %.4f, update %.2f ms per chunk\n", ...
        R.incTable.Accuracy(1), R.incTable.Accuracy(end), R.accFullRetrain, ...
        R.accFullRetrainRF, R.updateMsPerChunk);
    showT(R.incTable);
    p("STEP 7b TRANSFER ACROSS BANDS\n"); showT(R.transferTable);
    p("STEP 7c LOW SNR\n"); showT(R.lowTable);
    p("STEP 7d REAL TIME (frame %.4f ms)\n", R.frameMs); showT(R.rtTable);
end
x = fullfile(cfg.resultsDir, "p2_xgboost_results.txt");
if isfile(x)
    p("OPTIONAL PYTHON XGBOOST\n%s\n", strjoin(readlines(x), newline));
end
end

function localShow(fid, t)
txt = evalc("disp(t)");
fprintf(fid, "%s\n", txt);
end

function localSection(p, showT, csvFile, title)
if isfile(csvFile)
    p("%s\n", title);
    showT(readtable(csvFile));
end
end

% -------------------------------------------------------------------------
function zipFile = localPack(cfg)
diary("off");
zipFile = fullfile(cfg.rootDir, "Project2_results_to_send.zip");
r = dir(fullfile(cfg.resultsDir, "*.*"));
r = r(~[r.isdir]);
items = "results/" + string({r.name});
extra = ["models/selection.mat" "datasets/timing.mat"];
for e = extra
    if isfile(fullfile(cfg.rootDir, e)), items(end+1) = e; end %#ok<AGROW>
end
try
    zip(zipFile, cellstr(items), cfg.rootDir);
catch err
    fprintf(2, "Could not create the zip file: %s\n", err.message);
end
diary(fullfile(cfg.resultsDir, "command_window_log.txt"));
end
