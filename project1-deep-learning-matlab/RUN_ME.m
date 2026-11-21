%% RUN_ME - runs the whole of Project 1 with one click
%
%   Open this file in MATLAB and press Run (or type RUN_ME in the Command
%   Window while this folder is the Current Folder).
%
%   It will:
%     1. check your MATLAB release, toolboxes and the ResNet-18 add-on
%     2. fetch the MathWorks example network for transfer learning
%     3. choose run settings to suit your computer (GPU or not)
%     4. run every handout step: 1, 2, 3.1, 3.2, 3.3, 3.4 and the overlap test
%     5. save every figure in the "results" folder, a log of everything
%        printed, a RESULTS_SUMMARY.txt file with the numbers for the report,
%        and Project1_results_to_send.zip containing all of it
%
%   Change MODE below only if you want to:
%     "auto"  - full run with a GPU, smaller run on a CPU (default)
%     "quick" - about 15-30 minutes, only to check that everything works
%     "full"  - full size even without a GPU (can take a day on a CPU)

MODE = "auto";

% Which step to start from (1 = everything). RUN_FINISH.m uses 5 to redo
% only validation, the over-the-air test and the overlap test, reusing the
% data set and the trained network that are already on disk.
START_STEP = 1;

if exist("MODE_OVERRIDE", "var"),  MODE = MODE_OVERRIDE;        end
if exist("START_OVERRIDE", "var"), START_STEP = START_OVERRIDE; end
clear MODE_OVERRIDE START_OVERRIDE

%% ------------------------------------------------------------- setup
clc;
close all;
% Figures are drawn hidden and saved straight to the results folder, so a
% sleeping screen or a closed window cannot interrupt the run.
oldVisible = get(groot, "DefaultFigureVisible");
set(groot, "DefaultFigureVisible", "off");
here = fileparts(mfilename("fullpath"));
cd(here);
addpath(here, fullfile(here,"helpers"));
if ~isfolder(fullfile(here,"results")), mkdir(fullfile(here,"results")); end
logFile = fullfile(here, "results", "command_window_log.txt");
if isfile(logFile), delete(logFile); end
diary(logFile);
tAll = tic;

fprintf("==============================================================\n");
fprintf(" Project 1: Deep-learning classification of Bluetooth and WiFi\n");
fprintf(" Started %s\n", string(datetime("now")));
fprintf("==============================================================\n\n");

%% ------------------------------------------------ 1. check the system
fprintf("[1/8] Checking MATLAB and toolboxes...\n");
if isMATLABReleaseOlderThan("R2024a")
    error("This project needs MATLAB R2024a or newer. You have %s.", version("-release"));
end
v = ver;
installed = string({v.Name});
required = ["Communications Toolbox" "Bluetooth Toolbox" ...
            "Signal Processing Toolbox" "DSP System Toolbox" "Deep Learning Toolbox" ...
            "Computer Vision Toolbox" "Image Processing Toolbox"];
missing = required(~ismember(required, installed));
if ~isempty(missing)
    error("These toolboxes are missing. Install them (Home > Add-Ons) and run again:\n  %s", ...
        strjoin(missing, newline + "  "));
end
fprintf("      MATLAB %s, all required toolboxes found.\n", version("-release"));
if hasWlanToolbox()
    fprintf("      WLAN Toolbox found: WiFi packets from wlanWaveformGenerator.\n");
else
    fprintf("      WLAN Toolbox not found: WiFi packets from the project's own\n");
    fprintf("      802.11 generator (helpers/wlanPacketFallback.m). Fine to continue.\n");
end

try
    imagePretrainedNetwork("resnet18");
    fprintf("      ResNet-18 model found.\n");
catch
    error(['The ResNet-18 model add-on is missing. In MATLAB go to Home > Add-Ons > ' ...
           'Get Add-Ons, search "Deep Learning Toolbox Model for ResNet-18 Network", ' ...
           'install it, and run RUN_ME again.']);
end

hasGPU = canUseGPU();
hasParallel = ismember("Parallel Computing Toolbox", installed);
fprintf("      GPU available: %s.   Parallel Computing Toolbox: %s.\n\n", ...
    string(hasGPU), string(hasParallel));

%% ------------------------------------------------ 2. choose run size
fprintf("[2/8] Choosing run settings (MODE = %s)...\n", MODE);
if MODE == "auto"
    if hasGPU
        MODE = "full";
    else
        MODE = "cpu";
    end
end
switch MODE
    case "quick"
        ov = struct("numFrames",100, "maxEpochs",3, "miniBatchSize",8, "learnRateDropPeriod",2);
    case "cpu"
        ov = struct("numFrames",600, "maxEpochs",10, "miniBatchSize",8, "learnRateDropPeriod",4);
    otherwise   % "full"
        ov = struct("numFrames",1500, "maxEpochs",20, "miniBatchSize",16, "learnRateDropPeriod",8);
end
setappdata(0, "RFP1_overrides", ov);
cfg = projectConfig();
fprintf("      Run size: %s  ->  %d frames, %d epochs, mini-batch %d.\n", ...
    upper(MODE), cfg.numFrames, cfg.maxEpochs, cfg.miniBatchSize);
if MODE == "cpu"
    fprintf("      (No GPU found, so a smaller run is used. Expect several hours.)\n");
end
fprintf("\n");

%% ------------------------------------- 3. get the example network
fprintf("[3/8] Getting the MathWorks example network for transfer learning...\n");
pretrainedOK = ensurePretrainedNetwork(cfg);
if pretrainedOK
    fprintf("      Found: %s\n\n", fullfile(cfg.pretrainedDir, cfg.pretrainedFile));
else
    fprintf(['      Could not get %s automatically.\n' ...
             '      Training will start from the ImageNet ResNet-18 backbone instead\n' ...
             '      (still transfer learning). See README section 3 to add the file.\n\n'], ...
             cfg.pretrainedFile);
end

%% ------------------------------------------------ 4. run the steps
steps = {
    "step1_generate_waveforms",  "Handout step 1: generate Bluetooth and WiFi",   true
    "step2_frequency_translate", "Handout step 2: Multiband Combiner",            true
    "step3a_create_dataset",     "Handout step 3.1: create the data set",         true
    "step3b_train_network",      "Handout step 3.2: transfer learning",           true
    "step3c_validate_network",   "Handout step 3.3: validate",                    true
    "step4_test_over_the_air",   "Handout step 3.4: over-the-air test",           false
    "advanced_overlap_test",     "Advanced: overlapping signals",                 false
    };
status = strings(size(steps,1),1);
for k = 1:size(steps,1)
    name = steps{k,1};
    if k < START_STEP
        status(k) = "skipped (already done)";
        continue
    end
    fprintf("[4/8] (%d/%d) %s\n", k, size(steps,1), steps{k,2});
    fprintf("--------------------------------------------------------------\n");
    tStep = tic;
    try
        runStep(name);
        status(k) = sprintf("OK (%.1f min)", toc(tStep)/60);
    catch err
        status(k) = "FAILED: " + string(err.message);
        fprintf(2, "\n*** %s failed: %s\n", name, err.message);
        for s = 1:min(3, numel(err.stack))
            fprintf(2, "    in %s (line %d)\n", err.stack(s).name, err.stack(s).line);
        end
        if steps{k,3}
            fprintf(2, "*** This step is needed by the next ones, so the run stops here.\n");
            fprintf(2, "*** Send results/command_window_log.txt to get it fixed.\n\n");
            writeSummary(cfg, steps, status, MODE, pretrainedOK, toc(tAll));
            packResults(cfg);
            set(groot, "DefaultFigureVisible", oldVisible);
            return
        end
    end
    fprintf("\n");
    drawnow;
end

%% ------------------------------------- 5-8. summary and packaging
fprintf("[5/8] Writing results/RESULTS_SUMMARY.txt...\n");
writeSummary(cfg, steps, status, MODE, pretrainedOK, toc(tAll));
fprintf("[6/8] Packing results...\n");
zipFile = packResults(cfg);
fprintf("[7/8] Step status:\n");
for k = 1:size(steps,1)
    fprintf("      %-28s %s\n", steps{k,1}, status(k));
end
fprintf("[8/8] Finished in %.1f minutes.\n\n", toc(tAll)/60);
fprintf("==============================================================\n");
fprintf(" All figures:       %s\n", cfg.resultsDir);
fprintf(" Numbers for report: %s\n", fullfile(cfg.resultsDir,"RESULTS_SUMMARY.txt"));
fprintf(" Send this file:    %s\n", zipFile);
fprintf("==============================================================\n");
set(groot, "DefaultFigureVisible", oldVisible);
diary("off");


%% =================================================================
%% local functions
%% =================================================================
function runStep(name)
% Run a step script in its own workspace so that variables do not clash.
run(char(name) + ".m");
end


% -------------------------------------------------------------------------
function ok = ensurePretrainedNetwork(cfg)
% Puts specSenseTrainedNetResnet18.mat in the pretrained folder if it can.
target = fullfile(cfg.pretrainedDir, cfg.pretrainedFile);
if ~isfolder(cfg.pretrainedDir), mkdir(cfg.pretrainedDir); end
ok = isfile(target);
if ok, return; end

% (a) already on the MATLAB path?
onPath = which(cfg.pretrainedFile);
if ~isempty(onPath)
    copyfile(onPath, target);
    ok = true;
    return
end

% (b) open the MathWorks example and use its own download helper
startDir = pwd;
restore = onCleanup(@() cd(startDir));
ids = ["comm/SpectrumSensingWithDeepLearningToIdentify5GAndLTESignalsExample"
       "5g/SpectrumSensingWithDeepLearningToIdentify5GAndLTESignalsExample"
       "comm/SpectrumSensingWithDeepLearningExample"
       "5g/SpectrumSensingWithDeepLearning5GLTEExample"];
for i = 1:numel(ids)
    try
        openExample(ids(i));             % copies the example and cd's into it
        exDir = pwd;
        addpath(exDir);
        if exist("helperSpecSenseDownloadDataAndNetworks","file")
            fprintf("      Downloading with the example's helper (may take a few minutes)...\n");
            helperSpecSenseDownloadDataAndNetworks("Generated data", false, false, ...
                "resnet18", [256 256]);
        end
        found = dir(fullfile(exDir, "**", cfg.pretrainedFile));
        if isempty(found)
            w = which(cfg.pretrainedFile);
            if ~isempty(w), found = dir(w); end
        end
        cd(startDir);
        if ~isempty(found)
            copyfile(fullfile(found(1).folder, found(1).name), target);
            ok = true;
            return
        end
    catch err
        cd(startDir);
        fprintf("      (example id %s: %s)\n", ids(i), err.message);
    end
end
end

% -------------------------------------------------------------------------
function writeSummary(cfg, steps, status, mode, pretrainedOK, seconds)
f = fullfile(cfg.resultsDir, "RESULTS_SUMMARY.txt");
fid = fopen(f, "w");
closer = onCleanup(@() fclose(fid));
p = @(varargin) fprintf(fid, varargin{:});

p("PROJECT 1 - RESULTS SUMMARY\n");
p("Generated %s, MATLAB %s, run size %s, total time %.1f min\n\n", ...
    string(datetime("now")), version("-release"), upper(mode), seconds/60);
p("Settings: %d frames (train/val/test %s), %d epochs, mini-batch %d, GPU %s\n", ...
    cfg.numFrames, mat2str(round(cfg.numFrames*cfg.split)), cfg.maxEpochs, ...
    cfg.miniBatchSize, string(canUseGPU()));
p("Example network file found: %s\n", string(pretrainedOK));
p("WiFi generator: %s\n\n", ternary(hasWlanToolbox(), "WLAN Toolbox", ...
    "own 802.11 generator (WLAN Toolbox not available)"));

p("STEP STATUS\n");
for k = 1:size(steps,1)
    p("  %-28s %s\n", steps{k,1}, status(k));
end
p("\n");

% ---- step 2
w = fullfile(cfg.waveformDir, "combinedWaveform.mat");
if isfile(w)
    W = load(w, "offsets", "fs");
    p("STEP 2 (report Table 10)\n  Combiner output rate: %.0f MHz\n", W.fs/1e6);
    p("  Offsets: Bluetooth %+.1f MHz, WLAN %+.1f MHz\n", W.offsets/1e6);
    p("  Bluetooth occupies %+.1f to %+.1f MHz\n", (W.offsets(1)+[-0.5 0.5]*1e6)/1e6);
    p("  WLAN occupies %+.1f to %+.1f MHz\n", (W.offsets(2)+[-8.9 8.9]*1e6)/1e6);
    p("  Separation: %.1f MHz (no overlap)\n\n", (W.offsets(2)-8.9e6 - (W.offsets(1)+0.5e6))/1e6);
end

% ---- class balance and training
if isfile(cfg.networkFile)
    S = load(cfg.networkFile);
    try
        p("STEP 3.1 CLASS WEIGHTS (report Table 11)\n");
        for i = 1:numel(S.classNames)
            p("  %-10s weight %.4f\n", S.classNames(i), S.classWeights(i));
        end
        lblDir = fullfile(cfg.dataDir, "train", "labels");
        pxds = pixelLabelDatastore(lblDir, cfg.classNames, cfg.labelIDs, FileExtensions=".png");
        T = countEachLabel(pxds);
        for i = 1:height(T)
            p("  %-10s pixels %d  (%.2f %% of all training pixels)\n", string(T.Name(i)), ...
                T.PixelCount(i), 100*T.PixelCount(i)/sum(T.PixelCount));
        end
        p("\n");
    catch err
        p("  (could not read class balance: %s)\n\n", err.message);
    end
    try
        p("STEP 3.2 TRAINING (report Table 12)\n");
        p("  Starting point: %s\n", S.report.source);
        p("  Parameter arrays copied: %d of %d\n", S.report.numCopied, S.report.numParameters);
        th = S.trainInfo.TrainingHistory;
        vh = S.trainInfo.ValidationHistory;
        if ismember("Epoch", string(th.Properties.VariableNames))
            p("  Epochs completed: %d\n", max(th.Epoch));
        end
        tl = lossColumn(th);
        if ~isempty(tl), p("  Final training loss: %.4f\n", tl(end)); end
        vl = lossColumn(vh);
        if ~isempty(vl), p("  Best validation loss: %.4f\n", min(vl)); end
        p("\n");
    catch err
        p("  (could not read training history: %s)\n\n", err.message);
    end
end

% ---- validation metrics
m = fullfile(cfg.resultsDir, "step3c_metrics.mat");
if isfile(m)
    M = load(m);
    d0 = dir(m); p("(file saved %s)\n", d0.date);
    d = M.metrics.DataSetMetrics;
    p("STEP 3.3 VALIDATION ON HELD-OUT FRAMES (report Tables 13 and 14)\n");
    p("  Global accuracy %.4f   Mean accuracy %.4f   Mean IoU %.4f\n", ...
        d.GlobalAccuracy, d.MeanAccuracy, d.MeanIoU);
    c = M.metrics.ClassMetrics;
    rn = string(c.Properties.RowNames);
    for i = 1:height(c)
        p("  %-10s accuracy %.4f   IoU %.4f\n", rn(i), c.Accuracy(i), c.IoU(i));
    end
    p("\n");
end

% ---- over the air
o = dir(fullfile(cfg.resultsDir, "ota", "ota_*.mat"));
if ~isempty(o)
    p("STEP 3.4 OVER-THE-AIR (report Table 15)\n");
    for i = 1:numel(o)
        O = load(fullfile(o(i).folder, o(i).name));
        p("  (file saved %s)\n", o(i).date);
        p("  %s: %s\n", o(i).name, O.source);
        s = O.summary;
        for j = 1:height(s.occupancy)
            p("     %-10s %.2f %% of time-frequency plane\n", ...
                s.occupancy.Class(j), s.occupancy.PercentOfTimeFrequency(j));
        end
        for j = 1:height(s.wlanBands)
            p("     WLAN channel at %.1f MHz, %.1f MHz wide, airtime %.0f %%\n", ...
                s.wlanBands.CenterMHz(j), s.wlanBands.WidthMHz(j), s.wlanBands.AirtimePercent(j));
        end
        if ~isnan(s.btBursts)
            p("     Bluetooth bursts: %d at about %s MHz\n", s.btBursts, mat2str(s.btChannels));
        end
    end
    p("\n");
end

% ---- overlap
a = fullfile(cfg.resultsDir, "advanced_overlap_metrics.mat");
if isfile(a)
    A = load(a);
    d0 = dir(a); p("(file saved %s)\n", d0.date);
    c = A.metricsOverlap.ClassMetrics;
    rn = string(c.Properties.RowNames);
    p("ADVANCED: OVERLAPPING SIGNALS (report Table 16)\n");
    p("  Mean IoU %.4f\n", A.metricsOverlap.DataSetMetrics.MeanIoU);
    for i = 1:height(c)
        p("  %-10s IoU %.4f\n", rn(i), c.IoU(i));
    end
end
end

% -------------------------------------------------------------------------
function zipFile = packResults(cfg)
diary("off");
zipFile = fullfile(cfg.rootDir, "Project1_results_to_send.zip");
items = {};
r = dir(fullfile(cfg.resultsDir, "**", "*.*"));
r = r(~[r.isdir]);
for i = 1:numel(r)
    if ~contains(r(i).folder, "test_predictions") && ~contains(r(i).folder, "overlap_predictions")
        full = fullfile(r(i).folder, r(i).name);
        items{end+1} = extractAfter(string(full), string(cfg.rootDir) + filesep); %#ok<AGROW>
    end
end
try
    zip(zipFile, cellstr(string(items)), cfg.rootDir);
catch err
    fprintf(2, "Could not create the zip file: %s\n", err.message);
end
diary(fullfile(cfg.resultsDir, "command_window_log.txt"));
end

% -------------------------------------------------------------------------
function v = lossColumn(tbl)
% Loss values from a training or validation history table.
v = [];
if ~istable(tbl) || isempty(tbl), return; end
names = string(tbl.Properties.VariableNames);
k = find(contains(names, "Loss", "IgnoreCase", true), 1);
if isempty(k), return; end
v = tbl.(names(k));
v = v(~isnan(v));
end

% -------------------------------------------------------------------------
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
