function w = p2_wlanPacketOwn(format, rate, psduBytes)
%P2_WLANPACKETOWN  One WiFi packet without WLAN Toolbox.
%
%   w = p2_wlanPacketOwn("nonHT", mcs, psduBytes)   802.11a/g OFDM, 20 MHz
%   w = p2_wlanPacketOwn("HT",    mcs, psduBytes)   802.11n HT mixed format, 20 MHz
%   w = p2_wlanPacketOwn("DSSS",  rateMbps, psduBytes)  802.11b, 11 Msample/s (one sample per chip)
%
%   Used only when WLAN Toolbox is not available. The packets follow the
%   802.11 physical-layer structure that matters for a spectrogram:
%   preambles, header fields, subcarrier layout (52 or 56 used of 64),
%   pilots, 4 us OFDM symbols with 0.8 us cyclic prefix, modulation per
%   MCS, and the packet duration that follows from the data rate. The data
%   bits are random, so scrambling and channel coding (which do not change
%   how the signal looks) are not modelled. Output has unit average power.

switch string(format)
    case "nonHT"
        w = localOFDM(false, rate, psduBytes);
    case "HT"
        w = localOFDM(true, rate, psduBytes);
    case "DSSS"
        w = localDSSS(rate, psduBytes);
    otherwise
        error("p2_wlanPacketOwn: unknown format %s.", format);
end
w = w(:);
w = w/sqrt(mean(abs(w).^2));
end

% =========================================================================
function w = localOFDM(isHT, mcs, L)
% Legacy short training field (802.11a, subcarriers -26..26)
S = sqrt(13/6)*[0 0 1+1j 0 0 0 -1-1j 0 0 0 1+1j 0 0 0 -1-1j 0 0 0 -1-1j ...
    0 0 0 1+1j 0 0 0 0 0 0 0 -1-1j 0 0 0 -1-1j 0 0 0 1+1j 0 0 0 1+1j ...
    0 0 0 1+1j 0 0 0 1+1j 0 0];
% Legacy long training field (subcarriers -26..26)
Lt = [1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 0 ...
      1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1];

stf = localIFFT(S, -26:26);
stf = [stf; stf; stf(1:32)];                      % 160 samples, 8 us
ltf = localIFFT(Lt, -26:26);
ltf = [ltf(33:64); ltf; ltf];                     % 160 samples, 8 us
w = [stf; ltf];

legacyData = setdiff(-26:26, [0 -21 -7 7 21]);    % 48 data subcarriers
w = [w; localSymbol(localQAM(2,48), legacyData)];                 % L-SIG

if isHT
    w = [w; localSymbol(1j*localQAM(2,48), legacyData)];          % HT-SIG 1 (QBPSK)
    w = [w; localSymbol(1j*localQAM(2,48), legacyData)];          % HT-SIG 2 (QBPSK)
    htStf = localIFFT(S, -26:26);
    w = [w; htStf; htStf(1:16)];                                  % HT-STF, 4 us
    htSc = setdiff(-28:28, 0);
    htLtf = localIFFT(sign(randn(1,56)) + 0, htSc);               % HT-LTF, 4 us
    w = [w; htLtf(49:64); htLtf];
    ndbps = [26 52 78 104 156 208 234 260];
    order = [2 4 4 16 16 64 64 64];
    dataSc = setdiff(-28:28, [0 -21 -7 7 21]);                    % 52 data
else
    ndbps = [24 36 48 72 96 144 192 216];
    order = [2 2 4 4 16 16 64 64];
    dataSc = legacyData;                                          % 48 data
end
nSym = ceil((16 + 8*L + 6)/ndbps(mcs+1));
for s = 1:nSym
    w = [w; localSymbol(localQAM(order(mcs+1), numel(dataSc)), dataSc)]; %#ok<AGROW>
end
end

% -------------------------------------------------------------------------
function t = localSymbol(dataVals, dataSc)
% One 4 us OFDM symbol: data + 4 BPSK pilots at +/-7, +/-21, CP of 16.
pil = [-21 -7 7 21];
vals = [dataVals(:).' sign(randn)*[1 1 1 -1]];
t = localIFFT(vals, [dataSc pil]);
t = [t(49:64); t];
end

% -------------------------------------------------------------------------
function t = localIFFT(vals, k)
X = zeros(64,1);
X(mod(k,64) + 1) = vals;
t = ifft(X)*64/sqrt(nnz(X));
end

% -------------------------------------------------------------------------
function s = localQAM(M, n)
if M == 2
    s = sign(randn(n,1));
    return
end
m = sqrt(M);
a = 2*randi([0 m-1], n, 1) - (m-1);
b = 2*randi([0 m-1], n, 1) - (m-1);
s = (a + 1j*b)/sqrt(2*(M-1)/3);
end

% =========================================================================
function w = localDSSS(rate, L)
% Long preamble (128 sync + 16 SFD bits) and 48-bit header at 1 Mb/s
% DBPSK with the 11-chip Barker code, then the payload at the chosen rate.
barker = [1 -1 1 1 -1 1 1 1 -1 -1 -1];
hdr = exp(1j*pi*cumsum(randi([0 1],192,1)));
w = kron(hdr, barker(:));
nBits = 8*L;
switch rate
    case 1                                       % DBPSK, 11 chips per bit
        s = exp(1j*pi*cumsum(randi([0 1],nBits,1)));
        w = [w; kron(s, barker(:))];
    case 2                                       % DQPSK, 11 chips per 2 bits
        s = exp(1j*pi/2*cumsum(randi([0 3],ceil(nBits/2),1)));
        w = [w; kron(s, barker(:))];
    otherwise                                    % CCK, 8 QPSK chips per symbol
        bitsPerSym = 4*(rate == 5.5) + 8*(rate == 11);
        nChips = 8*ceil(nBits/bitsPerSym);
        w = [w; exp(1j*pi/2*randi([0 3],nChips,1))];
end
end
