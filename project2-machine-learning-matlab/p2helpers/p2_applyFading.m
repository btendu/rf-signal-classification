function [y, name] = p2_applyFading(x, fs)
%P2_APPLYFADING  Random multipath fading channel (Communications Toolbox).
%
%   [y, name] = p2_applyFading(x, fs)
%
%   One of four standard profiles, each with a random Doppler of 0-50 Hz:
%     Flat Rician (K = 0-10 dB)          - line of sight, no delay spread
%     Indoor 4-tap (0-150 ns)            - office / home
%     EPA (Extended Pedestrian A, 410 ns)  - 3GPP
%     EVA (Extended Vehicular A, 2.5 us)   - 3GPP

fd = 50*rand;
switch randi(4)
    case 1
        name = "Flat Rician";
        ch = comm.RicianChannel('SampleRate',fs, 'KFactor',10^(rand), ...
            'PathDelays',0, 'AveragePathGains',0, 'MaximumDopplerShift',fd);
    case 2
        name = "Indoor 4-tap";
        ch = comm.RayleighChannel('SampleRate',fs, ...
            'PathDelays',[0 50 100 150]*1e-9, 'AveragePathGains',[0 -3 -6 -9], ...
            'NormalizePathGains',true, 'MaximumDopplerShift',fd);
    case 3
        name = "EPA";
        ch = comm.RayleighChannel('SampleRate',fs, ...
            'PathDelays',[0 30 70 90 110 190 410]*1e-9, ...
            'AveragePathGains',[0 -1 -2 -3 -8 -17.2 -20.8], ...
            'NormalizePathGains',true, 'MaximumDopplerShift',fd);
    otherwise
        name = "EVA";
        ch = comm.RayleighChannel('SampleRate',fs, ...
            'PathDelays',[0 30 150 310 370 710 1090 1730 2510]*1e-9, ...
            'AveragePathGains',[0 -1.5 -1.4 -3.6 -0.6 -9.1 -7 -12 -16.9], ...
            'NormalizePathGains',true, 'MaximumDopplerShift',fd);
end
y = ch(double(x(:)));
end
