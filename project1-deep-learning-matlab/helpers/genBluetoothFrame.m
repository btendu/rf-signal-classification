function [x, boxes] = genBluetoothFrame(cfg, hopOffsets, activity)
%GENBLUETOOTHFRAME  One frame of frequency-hopping Bluetooth BR/EDR packets.
%
%   [x, boxes] = genBluetoothFrame(cfg, hopOffsets, activity)
%
%   x      N-by-K complex single matrix at cfg.btRate. Column k holds every
%          packet sent on hop frequency hopOffsets(k). All columns are at
%          baseband; the Multiband Combiner moves each one to its hop.
%   boxes  one row per packet: [tStart tEnd fLow fHigh] in seconds and Hz
%          (frequencies relative to the centre of the captured band).
%
%   activity is the chance that a 625 us slot starts a packet.

fsBT    = cfg.btRate;
N       = round(cfg.frameDuration*fsBT);
K       = numel(hopOffsets);
slotLen = round(625e-6*fsBT);

x     = complex(zeros(N,K,"single"));
boxes = zeros(0,4);

% Flat Rician fading, one complex gain per hop frequency. A 1 MHz hop is
% much narrower than the indoor coherence bandwidth, so each hop simply
% arrives stronger or weaker.
kFactor = 10^(10*rand/10);                       % 0 to 10 dB
gain = sqrt(kFactor/(kFactor+1)) + ...
    sqrt(1/(2*(kFactor+1)))*(randn(1,K) + 1j*randn(1,K));

pos = randi(slotLen);            % the capture is not aligned to the slots
while pos <= N - round(100e-6*fsBT)
    if rand < activity
        w = localRandomPacket(cfg.btSamplesPerSymbol);
        nActive = find(abs(w) > 0, 1, "last");   % generator pads with zeros
        if isempty(nActive)
            pos = pos + slotLen;
            continue
        end
        k  = randi(K);
        n1 = min(N, pos + nActive - 1);
        x(pos:n1,k) = single(gain(k)*w(1:n1-pos+1));
        boxes(end+1,:) = [(pos-1)/fsBT, n1/fsBT, ...
            hopOffsets(k) - cfg.btBandwidth/2, ...
            hopOffsets(k) + cfg.btBandwidth/2]; %#ok<AGROW>
        pos = pos + ceil(numel(w)/slotLen)*slotLen;   % 1, 3 or 5 slots
    else
        pos = pos + slotLen;
    end
end
end

% -------------------------------------------------------------------------
function w = localRandomPacket(sps)
% Random ACL data packet: basic rate (GFSK) or enhanced data rate (DPSK),
% occupying 1, 3 or 5 slots, with a random payload size.
modes      = ["BR" "BR" "EDR2M" "EDR3M"];     % basic rate twice as likely
types      = ["DH1" "DH3" "DH5"];
maxPayload = [27 183 339;                     % BR     (bytes)
              54 367 679;                     % EDR 2M
              83 552 1021];                   % EDR 3M
phyMode = modes(randi(numel(modes)));
ti   = randi(numel(types));
mx   = maxPayload(find(phyMode == ["BR" "EDR2M" "EDR3M"],1), ti);
try
    c = bluetoothWaveformConfig(Mode=phyMode, PacketType=types(ti), ...
        SamplesPerSymbol=sps);
    c.PayloadLength = randi([ceil(0.3*mx) mx]);
    bits = randi([0 1], getPayloadLength(c)*8, 1);
    w = bluetoothWaveformGenerator(bits, c);
catch err
    % Packet-type naming for EDR differs between releases. Fall back to a
    % basic-rate packet rather than stopping a long data-set run.
    warning("rfproject:btFallback", ...
        "Bluetooth %s/%s not available (%s). Using a BR packet instead.", ...
        phyMode, types(ti), err.message);
    c = bluetoothWaveformConfig(Mode="BR", PacketType=types(ti), ...
        SamplesPerSymbol=sps);
    c.PayloadLength = maxPayload(1,ti);
    bits = randi([0 1], getPayloadLength(c)*8, 1);
    w = bluetoothWaveformGenerator(bits, c);
end
w = double(w(:));
end
