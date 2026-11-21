function [x, bw] = p2_genOFDMLike(scs, nRB, fs, duration)
%P2_GENOFDMLIKE  OFDM signal with LTE/NR numerology (fallback only).
%
%   [x, bw] = p2_genOFDMLike(scs, nRB, fs, duration)
%
%   Used only when LTE Toolbox or 5G Toolbox is missing. It has the right
%   subcarrier spacing, cyclic prefix and bandwidth, filled with random
%   QAM symbols, but no reference signals or control channels.

nfft = round(fs/scs);
K    = nRB*12;
bw   = K*scs;
symPerHalfMs = round(7*scs/15e3);
cpLong  = round(160*nfft/2048);
cpShort = round(144*nfft/2048);
M = [4 16 64];
m = M(randi(3));

kIdx = [(-K/2:-1) (1:K/2)];           % DC left empty
bins = mod(kIdx, nfft) + 1;

x = zeros(0,1);
s = 0;
while numel(x) < round(duration*fs)
    a = (randi([0 sqrt(m)-1],K,1)*2 - (sqrt(m)-1)) + ...
        1j*(randi([0 sqrt(m)-1],K,1)*2 - (sqrt(m)-1));
    X = zeros(nfft,1);
    X(bins) = a;
    t = ifft(X)*sqrt(nfft);
    if mod(s, symPerHalfMs) == 0
        cp = cpLong;
    else
        cp = cpShort;
    end
    x = [x; t(end-cp+1:end); t];      %#ok<AGROW>
    s = s + 1;
end
x = x/sqrt(mean(abs(x).^2));
end
