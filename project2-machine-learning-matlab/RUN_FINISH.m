%% RUN_FINISH - continue a full run that stopped part-way
%
%   Use this when RUN_ME stopped (error, power cut, MATLAB closed) after
%   some steps had finished. Everything already on disk is reused.
%
%   Set START_OVERRIDE to the step to start from:
%     2 = data sets   3 = feature selection   4 = model training
%     5 = robustness  6 = SDR validation      7 = advanced extensions
%   (Step 2 also reuses any data-set file that is already complete.)

MODE_OVERRIDE  = "full";
START_OVERRIDE = 5;
RUN_ME
