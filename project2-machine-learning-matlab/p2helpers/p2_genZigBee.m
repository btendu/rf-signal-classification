function e = p2_genZigBee(cfg)
%P2_GENZIGBEE  IEEE 802.15.4 (ZigBee) 2.4 GHz O-QPSK packets.
%
%   Written from the standard so that no add-on is needed:
%     bytes -> 4-bit symbols (low nibble first) -> 32-chip sequences
%     -> O-QPSK with half-sine pulses at 2 Mchip/s.
%   Each packet has the standard synchronisation header (4 zero bytes and
%   the start-of-frame byte 0xA7) and a length byte.

sps   = 8;                        % samples per chip -> 16 MHz
fsZ   = 2e6*sps;
total = round(cfg.poolDuration*fsZ);
chips = localChipTable();

x = zeros(0,1);
mask = false(0,1);
while numel(x) < total
    gap = round((100e-6 + rand*1.5e-3)*fsZ);
    len = randi([10 127]);
    bytes = [0 0 0 0 hex2dec('A7') len randi([0 255],1,len)];
    w = localModulate(bytes, chips, sps);
    x    = [x; zeros(gap,1); w];                     %#ok<AGROW>
    mask = [mask; false(gap,1); true(numel(w),1)];   %#ok<AGROW>
end

[e.x, e.mask] = p2_toCommonRate(x, mask, fsZ, cfg.fs);
e.bw      = 2e6;
e.subtype = "802.15.4 O-QPSK 250 kb/s";
e.source  = "Own 802.15.4 generator (p2_genZigBee)";
end

% -------------------------------------------------------------------------
function T = localChipTable()
% 16 symbols x 32 chips. Symbol 0 is given in the standard; symbols 1-7
% are cyclic shifts of it by 4 chips; symbols 8-15 are symbols 0-7 with
% their odd-indexed chips inverted.
c0 = '11011001110000110101001000101110' - '0';
T = zeros(16,32);
for k = 0:7
    T(k+1,:) = circshift(c0, 4*k);
end
for k = 8:15
    s = T(k-7,:);
    s(2:2:end) = 1 - s(2:2:end);
    T(k+1,:) = s;
end
end

% -------------------------------------------------------------------------
function w = localModulate(bytes, T, sps)
syms = zeros(1, 2*numel(bytes));
syms(1:2:end) = mod(bytes,16);          % low nibble first
syms(2:2:end) = floor(bytes/16);
c = reshape(T(syms+1,:).', 1, []);      % chip stream
d = 2*c - 1;
I = d(1:2:end).';                       % even chips on I
Q = d(2:2:end).';                       % odd chips on Q
p = sin(pi*(0:2*sps-1)'/(2*sps));       % half-sine, two chips long
up = zeros(numel(I)*2*sps,1);
up(1:2*sps:end) = I;
si = filter(p, 1, [up; zeros(2*sps,1)]);
up(:) = 0;
up(1:2*sps:end) = Q;
sq = filter(p, 1, [up; zeros(2*sps,1)]);
w = [si; zeros(sps,1)] + 1j*[zeros(sps,1); sq];   % Q delayed by one chip
w = w/sqrt(mean(abs(w).^2));
end
