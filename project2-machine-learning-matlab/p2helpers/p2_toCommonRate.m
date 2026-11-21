function [y, m] = p2_toCommonRate(x, mask, fsIn, fsOut)
%P2_TOCOMMONRATE  Resample a waveform and its activity mask to fsOut.
%
%   [y, m] = p2_toCommonRate(x, mask, fsIn, fsOut)

x = double(x(:));
mask = logical(mask(:));
if abs(fsIn - fsOut) < 1
    y = x;
    m = mask;
    return
end
[p, q] = rat(fsOut/fsIn, 1e-9);
y = resample(x, p, q);
m = resample(double(mask), p, q) > 0.5;
n = min(numel(y), numel(m));
y = y(1:n);
m = m(1:n);
end
