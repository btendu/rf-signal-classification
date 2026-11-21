%% Run the whole project in order
% Steps 1 to 3.3 need only MATLAB and the toolboxes listed in the README.
% Step 3.4 uses an SDR if one is connected.
%
% Each step can also be run on its own; later steps read the files that
% earlier steps saved.

step1_generate_waveforms      % handout step 1
step2_frequency_translate     % handout step 2
step3a_create_dataset         % handout step 3.1  (slow: builds 1500 frames)
step3b_train_network          % handout step 3.2  (slow: needs a GPU to be quick)
step3c_validate_network       % handout step 3.3
step4_test_over_the_air       % handout step 3.4
