%% STEP 1 - Generate the two signals (Bluetooth and WLAN)
% Handout: "Use MATLAB's Wireless Waveform Generator App to generate two of
% the signals above."
%
% There are two ways to do this step, and both end with the same two files:
%
%   A. In the app (see README, "Step 1 in the app"). Export the two
%      waveforms and save them as
%          waveforms/bluetooth_app.mat   and   waveforms/wlan_app.mat
%      This script then loads them.
%
%   B. Let this script generate them. It calls bluetoothWaveformGenerator
%      and wlanWaveformGenerator, the same functions the app runs when you
%      press Generate (the app's "Export MATLAB Script" produces code built
%      on these calls).
%
% Output: waveforms/bluetoothWaveform.mat and waveforms/wlanWaveform.mat,
% each holding a struct with fields  waveform  (complex samples) and  Fs.

cfg = projectConfig();
if ~isfolder(cfg.waveformDir), mkdir(cfg.waveformDir); end
rng(cfg.seed, "twister");

btAppFile   = fullfile(cfg.waveformDir, "bluetooth_app.mat");
wlanAppFile = fullfile(cfg.waveformDir, "wlan_app.mat");

%% Bluetooth BR/EDR
if isfile(btAppFile)
    [bt.waveform, bt.Fs] = loadAppWaveform(btAppFile, 20e6);
    bt.type = "Bluetooth BR/EDR (from the app)";
else
    % Four DH5 packets. Each one fills 5 slots (3.125 ms), so the waveform
    % lasts 12.5 ms. 20 samples per symbol gives a 20 MHz sample rate, the
    % same as the WLAN waveform, which is what the Multiband Combiner needs.
    numPackets = 4;
    btCfg = bluetoothWaveformConfig(Mode="BR", PacketType="DH5", ...
        SamplesPerSymbol=20);
    btCfg.PayloadLength = 339;                          % bytes, DH5 maximum
    bits = randi([0 1], getPayloadLength(btCfg)*8*numPackets, 1);
    bt.waveform = double(bluetoothWaveformGenerator(bits, btCfg));
    bt.Fs   = btCfg.SamplesPerSymbol*1e6;               % 1 Msym/s
    bt.type = "Bluetooth BR, DH5 packets";
end

%% WLAN
if isfile(wlanAppFile)
    [wlan.waveform, wlan.Fs] = loadAppWaveform(wlanAppFile, 20e6);
    wlan.type = "WLAN (from the app)";
else
    % Eight 802.11n (HT) packets on a 20 MHz channel with 0.5 ms gaps.
    if hasWlanToolbox()
        wlanCfg = wlanHTConfig(ChannelBandwidth="CBW20", MCS=4, PSDULength=2000);
        bits = randi([0 1], 8*wlanCfg.PSDULength, 1);
        wlan.waveform = wlanWaveformGenerator(bits, wlanCfg, ...
            NumPackets=8, IdleTime=0.5e-3);
        wlan.Fs   = wlanSampleRate(wlanCfg);
        wlan.type = "WLAN 802.11n HT, 20 MHz";
    else
        % Same packets with the project's own 802.11 generator
        % (WLAN Toolbox not available).
        idle = zeros(round(0.5e-3*20e6), 1);
        wlan.waveform = [];
        for k = 1:8
            wlan.waveform = [wlan.waveform; wlanPacketFallback("HT", 4, 2000); idle];
        end
        wlan.Fs   = 20e6;
        wlan.type = "WLAN 802.11n HT, 20 MHz (own generator, no WLAN Toolbox)";
    end
end

%% Save
waveformStruct = bt;   %#ok<NASGU>
save(fullfile(cfg.waveformDir,"bluetoothWaveform.mat"), "-struct", "waveformStruct");
waveformStruct = wlan; %#ok<NASGU>
save(fullfile(cfg.waveformDir,"wlanWaveform.mat"), "-struct", "waveformStruct");

fprintf("Bluetooth: %s, %.2f ms at %.0f MHz\n", bt.type, ...
    numel(bt.waveform)/bt.Fs*1e3, bt.Fs/1e6);
fprintf("WLAN:      %s, %.2f ms at %.0f MHz\n", wlan.type, ...
    numel(wlan.waveform)/wlan.Fs*1e3, wlan.Fs/1e6);

%% Look at them
figure("Color","w", "Name","Step 1 - generated waveforms");
tiledlayout(2,2, "TileSpacing","compact");

nexttile; plot((0:numel(bt.waveform)-1)/bt.Fs*1e3, abs(bt.waveform));
title("Bluetooth - amplitude"); xlabel("Time (ms)"); grid on
nexttile; pwelch(bt.waveform, [], [], 2048, bt.Fs, "centered");
title("Bluetooth - spectrum");

nexttile; plot((0:numel(wlan.waveform)-1)/wlan.Fs*1e3, abs(wlan.waveform));
title("WLAN - amplitude"); xlabel("Time (ms)"); grid on
nexttile; pwelch(wlan.waveform, [], [], 2048, wlan.Fs, "centered");
title("WLAN - spectrum");

if ~isfolder(cfg.resultsDir), mkdir(cfg.resultsDir); end
saveFigure(gcf, fullfile(cfg.resultsDir,"step1_waveforms.png"), Resolution=150);
