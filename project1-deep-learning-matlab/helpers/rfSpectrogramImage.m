function [img, PdB] = rfSpectrogramImage(x, imageSize, dynamicRangeDb)
%RFSPECTROGRAMIMAGE  Turn complex samples into the network's input image.
%
%   [img, PdB] = rfSpectrogramImage(x, imageSize, dynamicRangeDb)
%
%   x              complex baseband samples of one frame
%   imageSize      [rows columns], e.g. [256 256]
%   dynamicRangeDb span of the colour scale (default 50 dB)
%
%   img  rows-by-columns-by-3 uint8 RGB image
%        row 1      = start of the frame, last row = end of the frame
%        column 1   = -fs/2,  last column = just below +fs/2
%   PdB  the power in dB behind the image (rows-by-columns)
%
%   The same function is used for training data and for SDR captures, so
%   the network always sees images made the same way. Like the MathWorks
%   example it uses a Hann window and the parula colour map. The colour
%   scale is tied to the noise floor of each frame, which makes the image
%   independent of the receiver gain.

if nargin < 3
    dynamicRangeDb = 50;
end
R = imageSize(1);
C = imageSize(2);
x = double(x(:));

nfft = C;                        % one FFT bin per image column
hop  = nfft/2;                   % 50 % overlap
M    = floor((numel(x) - nfft)/hop) + 1;      % number of FFT slices
if M < R
    error("rfproject:frameTooShort", ...
        "Need at least %d samples for a %d-row image.", (R-1)*hop+nfft, R);
end
win   = 0.5 - 0.5*cos(2*pi*(0:nfft-1)'/nfft);   % Hann window
rowOf = min(R, floor((0:M-1)*R/M) + 1);         % image row of each slice

% Average the power of all slices that fall in the same image row.
P   = zeros(R,C);
cnt = zeros(R,1);
blockSize = 2048;                % slices per block, bounds memory use
for m0 = 1:blockSize:M
    m   = m0:min(M, m0+blockSize-1);
    idx = (0:nfft-1)' + (m-1)*hop + 1;          % nfft-by-numel(m)
    X   = fftshift(fft(x(idx).*win, [], 1), 1);
    A   = sparse(rowOf(m), 1:numel(m), 1, R, numel(m));
    P   = P + A*(abs(X).^2).';
    cnt = cnt + full(sum(A,2));
end
P   = P./max(cnt,1);
PdB = 10*log10(P + realmin);

% Colour scale: 5 dB below the noise floor up to dynamicRangeDb above that.
s = sort(PdB(:));
floorDb = s(max(1, round(0.10*numel(s))));      % 10th percentile
g = (PdB - (floorDb - 5))/dynamicRangeDb;
g = min(max(g,0),1);

idxImg = uint8(round(255*g));
img = uint8(round(255*ind2rgb(idxImg, parula(256))));
end
