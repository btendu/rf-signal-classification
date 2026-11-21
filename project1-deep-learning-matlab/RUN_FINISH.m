%% RUN_FINISH - finish a full run whose later steps stopped
%
%   Use this when RUN_ME (MODE = "full") already generated the data set and
%   trained the network, but stopped in step 3.3 or later.
%   It reuses the saved data and network (no retraining) and runs only:
%     step 3.3 validation, step 3.4 over-the-air test, the overlap test,
%   then writes RESULTS_SUMMARY.txt and Project1_results_to_send.zip.
%   Takes a few minutes.

MODE_OVERRIDE  = "full";
START_OVERRIDE = 5;
RUN_ME
