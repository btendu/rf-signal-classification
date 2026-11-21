function lbl = boxesToLabelImage(boxes, fs, T, imageSize)
%BOXESTOLABELIMAGE  Paint time-frequency boxes into a pixel-label image.
%
%   lbl = boxesToLabelImage(boxes, fs, T, imageSize)
%
%   boxes  one row per packet: [tStart tEnd fLow fHigh labelID]
%          times in seconds from the start of the frame, frequencies in Hz
%          relative to the centre of the band
%   fs     sample rate of the frame (Hz), T its duration (s)
%
%   lbl    uint8 image, 0 where nothing was painted. Later rows of boxes
%          are painted over earlier ones.
%
%   The geometry matches rfSpectrogramImage: row r covers the time
%   (r-1)*T/R to r*T/R, and column c is the FFT bin centred on
%   (c-1-C/2)*fs/C, which covers half a bin either side of that.

R = imageSize(1);
C = imageSize(2);
lbl = zeros(R,C,"uint8");

for i = 1:size(boxes,1)
    r0 = floor(boxes(i,1)/T*R) + 1;
    r1 = ceil(boxes(i,2)/T*R);
    u0 = (boxes(i,3) + fs/2)/fs*C;       % position in bins from -fs/2
    u1 = (boxes(i,4) + fs/2)/fs*C;
    c0 = floor(u0 + 0.5) + 1;
    c1 = ceil(u1 + 0.5);

    r0 = max(r0,1);  r1 = min(r1,R);
    c0 = max(c0,1);  c1 = min(c1,C);
    if r1 >= r0 && c1 >= c0
        lbl(r0:r1, c0:c1) = boxes(i,5);
    end
end
end
