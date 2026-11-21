function summary = summarizeSegmentation(labels, cfg, quiet)
%SUMMARIZESEGMENTATION  Say in words what the network found in one frame.
%
%   summary = summarizeSegmentation(labels, cfg)
%
%   summary.occupancy  share of the time-frequency plane given to each class
%   summary.wlanBands  centre and width (MHz) of each WLAN channel found
%   summary.btBursts   number of separate Bluetooth bursts
%   summary.btChannels their hop frequencies (MHz, rounded)

if nargin < 3
    quiet = false;
end
idx = labelsToIndex(labels, cfg);
[R, C] = size(idx);
binHz  = cfg.fs/C;

n   = numel(cfg.classNames);
pct = zeros(n,1);
for k = 1:n
    pct(k) = 100*mean(idx(:) == k);
end
summary.occupancy = table(cfg.classNames(:), pct, ...
    'VariableNames', ["Class" "PercentOfTimeFrequency"]);

% ---- WLAN: columns that are labelled WLAN for a visible share of the frame
iW = find(cfg.classNames == "WLAN");
colShare = mean(idx == iW, 1);
active = colShare > 0.03;
d  = diff([0 active 0]);
c0 = find(d == 1);
c1 = find(d == -1) - 1;
fLow  = cfg.centerFrequency + (c0 - 1 - C/2 - 0.5)*binHz;
fHigh = cfg.centerFrequency + (c1 - 1 - C/2 + 0.5)*binHz;
keep  = (fHigh - fLow) > 5e6;            % ignore slivers
airtime = zeros(1,numel(c0));
for k = 1:numel(c0)
    airtime(k) = 100*mean(any(idx(:,c0(k):c1(k)) == iW, 2));
end
summary.wlanBands = table((fLow(keep)+fHigh(keep)).'/2e6, ...
    (fHigh(keep)-fLow(keep)).'/1e6, airtime(keep).', ...
    'VariableNames', ["CenterMHz" "WidthMHz" "AirtimePercent"]);

% ---- Bluetooth: each connected blob is one burst
iB = find(cfg.classNames == "Bluetooth");
summary.btBursts   = NaN;
summary.btChannels = [];
if exist("bwconncomp","file") == 2
    cc = bwconncomp(idx == iB);
    minPixels = 6;                        % drop single-pixel speckle
    fc = zeros(0,1);
    for k = 1:cc.NumObjects
        if numel(cc.PixelIdxList{k}) >= minPixels
            [~, cols] = ind2sub([R C], cc.PixelIdxList{k});
            fc(end+1,1) = cfg.centerFrequency + (mean(cols) - 1 - C/2)*binHz; %#ok<AGROW>
        end
    end
    summary.btBursts   = numel(fc);
    summary.btChannels = unique(round(fc/1e6)).';
end

if ~quiet
    disp(summary.occupancy);
    if isempty(summary.wlanBands)
        fprintf("  No WLAN channel found.\n");
    else
        disp(summary.wlanBands);
    end
    if ~isnan(summary.btBursts)
        fprintf("  Bluetooth bursts: %d", summary.btBursts);
        if summary.btBursts > 0
            fprintf(" on about %s MHz", mat2str(summary.btChannels));
        end
        fprintf("\n");
    end
end
end
