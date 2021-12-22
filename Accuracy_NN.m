function [Pourcentage_detection_NN, Pourcentage_spurious_NN,Pourcentage_detection_LM, Pourcentage_spurious_LM] = Accuracy_NN(YPred,YTest,V_test,NT,dt,True_jump_times)
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Legend %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%

% YPred is a vector containing the jumps DETECTED by the NN

% YTest is a vector containing the TRUE jumps (known because I simulate them) (for returns)

% True_jump_times is a vector containing the TRUE jumps but for the price process (so it is one period shifted wrt to the one for returns)


%% Compute the accuracy of jump detection for the Neural Network (NN)

acc_total_NN = sum(YPred == YTest)./numel(YTest); % acc_total is the percentage of data correctly classified (for both jumps & no jumps)


pos_jump_YPred = find(YPred=='jump'); % Position on the x-axis (i.e. time) of the jumps DETECTED by the NN
pos_jump_YTest = find(YTest=='jump'); % Position of the TRUE jumps

% N.B: the positions of these jumps are for RETURNS (when there is a jump in the return, and not the price).
% So they are one period shifted from the time when there is a jump in the price.


Find_jump_times_union_NN = ismember(pos_jump_YPred,pos_jump_YTest);


F_j_t_u_2_NN = find(Find_jump_times_union_NN==1);
%Keep only the CORRECT jumps from Find_jump_times

Spurious_NN = find(Find_jump_times_union_NN==0);
%Keep only the SPURIOUS jumps from Find_jump_times


Pourcentage_detection_NN = length(F_j_t_u_2_NN)/length(pos_jump_YTest)*100;
% Pourcentage of CORRECTLY detected jumps
Pourcentage_spurious_NN = length(Spurious_NN)/length(pos_jump_YTest)*100;
% Pourcentage of SPURIOUSLY detected jumps


%% Compute the accuracy of jump detection for the Lee & Mykland 2008 (LM) test

% Detection Jumps (Lee Mykland 2008)

% Jumps detection Rate (ONLY on simulated data where I know the true number of jumps)
% I challenge the jump estimation on simulated data to assess the performance of the test


Stop_NT = length(V_test); %Length of the time series data

K_window = ceil(dt^(-0.52));% %Window size (as small as possible)
alpha_conf=0.01; % Confidence level




[Find_jump_times,~,~,~]=LM_jump_test(alpha_conf,K_window,V_test,Stop_NT);



Estimated_jump_times = zeros(NT,1);
Estimated_jump_times(Find_jump_times)=Find_jump_times;


% N.B: Find_jump_times are the jump times DETECTED by the Lee&Mykland test.
% True_jump_times are the TRUE jump times.
% Both of them are the jump times of the PRICE process (not the return).
% So they are shifted by one period wrt the jump times in the return process.


% This is to compute what is the % of jumps correctly detected (among all the jumps)

True_jump_times_filter = True_jump_times(True_jump_times~=0); %Get rid of the 0

Find_jump_times_union = ismember(Find_jump_times,True_jump_times_filter);
% Create a vector with 1 if the jump is correct and 0 if it's a spurious detection

F_j_t_u_2 = find(Find_jump_times_union==1);
%Keep only the CORRECT jumps from Find_jump_times

Spurious = find(Find_jump_times_union==0);
%Keep only the SPURIOUS jumps from Find_jump_times


Pourcentage_detection_LM = length(F_j_t_u_2)/length(True_jump_times_filter)*100;
% Pourcentage of CORRECTLY detected jumps
Pourcentage_spurious_LM = length(Spurious)/length(True_jump_times_filter)*100;
% Pourcentage of SPURIOUSLY detected jumps




end
