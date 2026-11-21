function p2_showFrame(ax, x, fs, titleText)
%P2_SHOWFRAME  Spectrogram of one frame in the given axes.
%
%   Frequency (MHz, relative to the band centre) left to right, time (us)
%   top to bottom, 50 dB colour range above the noise floor.

x = double(x(:));
n = 256;
hop = 64;
w = 0.5 - 0.5*cos(2*pi*(0:n-1)'/n);
M = floor((numel(x) - n)/hop) + 1;
idx = (0:n-1)' + (0:M-1)*hop + 1;
S = 10*log10(abs(fftshift(fft(x(idx).*w, [], 1), 1)).^2 + eps).';
s = sort(S(:));
fl = s(round(0.1*numel(s)));
f = ((0:n-1) - n/2)*fs/n/1e6;
t = ((0:M-1)*hop + n/2)/fs*1e6;
imagesc(ax, f, t, S, [fl - 5, fl + 45]);
colormap(ax, parula);
title(ax, titleText, "Interpreter", "none");
xlabel(ax, "Frequency (MHz)");
ylabel(ax, "Time (\mus)");
end
