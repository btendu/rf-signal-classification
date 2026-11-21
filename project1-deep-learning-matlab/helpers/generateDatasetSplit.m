function generateDatasetSplit(cfg, splitName, numFrames, seed, overlapFraction)
%GENERATEDATASETSPLIT  Write one part of the data set to disk.
%
%   generateDatasetSplit(cfg, splitName, numFrames, seed, overlapFraction)
%
%   Creates   <cfg.dataDir>/<splitName>/images/frame_00001.png ...
%             <cfg.dataDir>/<splitName>/labels/frame_00001.png ...
%             <cfg.dataDir>/<splitName>/meta.csv
%
%   Image and label files share a name, so the image datastore and the
%   pixel-label datastore line up. meta.csv records the random settings of
%   every frame (type, SNR, impairments).
%
%   Uses parfor: with Parallel Computing Toolbox the frames are generated
%   on all cores, without it the loop simply runs one frame at a time.

if nargin < 5
    overlapFraction = 0;
end
splitDir = fullfile(cfg.dataDir, splitName);
imgDir   = fullfile(splitDir, "images");
lblDir   = fullfile(splitDir, "labels");

existing = dir(fullfile(imgDir, "frame_*.png"));
if numel(existing) == numFrames && ~cfg.forceRegenerate ...
        && isfile(fullfile(splitDir,"meta.csv"))
    fprintf("  %-10s already has %d frames - skipped.\n", splitName, numFrames);
    return
end
if ~isfolder(imgDir), mkdir(imgDir); end
if ~isfolder(lblDir), mkdir(lblDir); end
% Remove frames from an earlier run of a different size, so the folders
% hold exactly the frames of this run.
delete(fullfile(imgDir, "frame_*.png"));
delete(fullfile(lblDir, "frame_*.png"));

% Decide the content of every frame up front so the result does not depend
% on how many parallel workers are used.
rng(seed, "twister");
typeNames = ["Bluetooth" "WLAN" "Both" "Noise"];
cumProb   = cumsum(cfg.frameTypeProb)/sum(cfg.frameTypeProb);
types     = typeNames(1 + sum(rand(numFrames,1) >= cumProb(1:end-1), 2));
allow     = rand(numFrames,1) < overlapFraction;

meta = cell(numFrames,1);
tStart = tic;
parfor i = 1:numFrames
    rng(seed + i, "twister");                 % each frame is repeatable
    [img, lbl, ~, m] = generateLabeledFrame(cfg, types(i), allow(i));
    name = sprintf("frame_%05d.png", i);
    imwrite(img, fullfile(imgDir, name));
    imwrite(lbl, fullfile(lblDir, name));
    m.file  = string(name);
    meta{i} = m;
    if mod(i,100) == 0
        fprintf("  %-10s frame %d done\n", splitName, i);
    end
end
writetable(struct2table([meta{:}]), fullfile(splitDir,"meta.csv"));
fprintf("  %-10s %d frames written in %.1f min.\n", ...
    splitName, numFrames, toc(tStart)/60);
end
