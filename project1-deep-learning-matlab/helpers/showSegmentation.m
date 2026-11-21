function fig = showSegmentation(img, cfg, titleText, varargin)
%SHOWSEGMENTATION  Spectrogram next to one or more label maps.
%
%   fig = showSegmentation(img, cfg, titleText, name1, labels1, ...)
%
%   Example:
%     showSegmentation(img, cfg, "Test frame 12", ...
%         "True labels", truth, "Predicted labels", predicted);
%
%   Frequency runs left to right in MHz, time runs top to bottom in ms.

numMaps = numel(varargin)/2;
R = size(img,1);
C = size(img,2);
f = (cfg.centerFrequency + ((0:C-1) - C/2)*cfg.fs/C)/1e6;
t = ((1:R) - 0.5)*cfg.frameDuration/R*1e3;

n = numel(cfg.classNames);
classColors = [0.13 0.13 0.16;     % Noise      (dark)
               0.20 0.65 1.00;     % Bluetooth  (blue)
               1.00 0.62 0.10;     % WLAN       (orange)
               0.40 0.85 0.40;     % spare colours for extra classes
               0.90 0.35 0.75];
classColors = classColors(1:n,:);

fig = figure("Color","w");
fig.Position(3:4) = [380*(numMaps+1) 430];
tl = tiledlayout(fig, 1, numMaps+1, "TileSpacing","compact", "Padding","compact");
title(tl, titleText, "Interpreter","none");

ax = nexttile(tl);
image(ax, f, t, img);
title(ax, "Spectrogram");
xlabel(ax, "Frequency (MHz)");
ylabel(ax, "Time (ms)");

for k = 1:numMaps
    ax = nexttile(tl);
    imagesc(ax, f, t, labelsToIndex(varargin{2*k}, cfg), [1 n]);
    colormap(ax, classColors);
    cb = colorbar(ax);
    cb.Ticks = 1 + ((1:n) - 0.5)*(n-1)/n;
    cb.TickLabels = cfg.classNames;
    title(ax, varargin{2*k-1});
    xlabel(ax, "Frequency (MHz)");
end
end
