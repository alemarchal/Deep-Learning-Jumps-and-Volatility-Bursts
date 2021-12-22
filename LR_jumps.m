function [lmd_plain , Regressor_total_RV_last , lmd_NN , Regressor_cont_RV_NN_last , lmd_LM , Regressor_cont_RV_LM_last , lmd_BPV , Regressor_BPV_last] = LR_jumps(fq,forecasting_horizons_in_days,pred_freq,total_RV_d, total_RV_w,total_RV_m,cont_RV_NN_d, cont_RV_NN_w,cont_RV_NN_m,JV_NN_d,JV_NN_w,JV_NN_m,cont_RV_LM_d,cont_RV_LM_w,cont_RV_LM_m,JV_LM_d,JV_LM_w,JV_LM_m,BPV_X_d,BPV_X_w,BPV_X_m,JV_BPV_d,JV_BPV_w,JV_BPV_m,incl_j_X)
% Linear Regression (LR) function
%% This classifies the returns into "jump"/"no jump".

% I should use this section if I want to make sure that I really don't use
% any information from the future (due to the bidirectional layer of the
% LSTM network).

% % Compute the Bipower Variation for the path
% K_window = ceil(dt^(-0.5));
% Bipower_StD=nan(length(K_window+1:length(V_test(:,1))),1);
% i=0;
% for ti=K_window+1:length(V_test) % Jump test time
%     i=i+1;
%     Bipower_StD(i) = BV(K_window,V_test,ti);
% end
% 
% r_ln_test = log(V_test(2:end) ./ V_test(1:end-1));%Vector of log-returns for the unique path
% 
% XTest = ( (r_ln_test(K_window:end,:)./Bipower_StD) .*fc)';
% YPred = classify(net,XTest);
% pos_jump_YPred = find(YPred=='jump')'+K_window-1;
% 
% Stop_NT = length(V_test); %Length of the time series data
% alpha_conf=0.01; % Confidence level for the LM test
% 
% [Find_jump_times_LM,~,~,~]=LM_jump_test(alpha_conf,K_window,V_test,Stop_NT);
% 
% 
% ind_jump_LM = zeros(length(V_test),1); 
% ind_jump_LM(Find_jump_times_LM)=1;
% ind_jump_LM=ind_jump_LM(2:end);
% ind_jump_NN = zeros(length(V_test)-1,1);
% ind_jump_NN(pos_jump_YPred)=1;
% 
% diff_ind_jump = ind_jump_LM + 10*ind_jump_NN;
% YDiff=categorical(diff_ind_jump(K_window:end)',[0 1 10 11],{'No jump both' 'Jump LM' 'Jump NN' 'Jump both'});


%% IN SAMPLE-Forecast of future TOTAL RV (we forecast continuous variance + jumps variance). 
% Because we don't care about forecasting only the continuous part because you can't trade only the continuous part.
% This in-sample estimation is just to get the coefficients

ahead=forecasting_horizons_in_days*fq; % in days
% Here I need to make sure we don't use future information for the forecast



if strcmp(incl_j_X,'yes') % Include the jump components as regressors

Regressor_total_RV= [total_RV_d(22*fq:end-ahead)' total_RV_w((22-4)*fq:end-ahead)' total_RV_m(1:end-ahead)'];

Regressor_cont_RV_NN= [cont_RV_NN_d(22*fq:end-ahead)'    cont_RV_NN_w((22-4)*fq:end-ahead)'   cont_RV_NN_m(1:end-ahead)' ...
                            JV_NN_d(22*fq:end-ahead)'         JV_NN_w((22-4)*fq:end-ahead)'        JV_NN_m(1:end-ahead)' ];                
                            
Regressor_cont_RV_LM= [cont_RV_LM_d(22*fq:end-ahead)'    cont_RV_LM_w((22-4)*fq:end-ahead)'    cont_RV_LM_m(1:end-ahead)' ...
                            JV_LM_d(22*fq:end-ahead)'         JV_LM_w((22-4)*fq:end-ahead)'         JV_LM_m(1:end-ahead)'];

Regressor_BPV= [BPV_X_d(22*fq:end-ahead)'    BPV_X_w((22-4)*fq:end-ahead)'    BPV_X_m(1:end-ahead)' ...
               JV_BPV_d(22*fq:end-ahead)'   JV_BPV_w((22-4)*fq:end-ahead)'   JV_BPV_m(1:end-ahead)' ];                     
          
elseif strcmp(incl_j_X,'no')  % Don't include the jump components                 
                        

Regressor_total_RV= [total_RV_d(22*fq:end-ahead)' total_RV_w((22-4)*fq:end-ahead)' total_RV_m(1:end-ahead)'];

Regressor_cont_RV_NN= [cont_RV_NN_d(22*fq:end-ahead)'    cont_RV_NN_w((22-4)*fq:end-ahead)'   cont_RV_NN_m(1:end-ahead)'];               
                            
Regressor_cont_RV_LM= [cont_RV_LM_d(22*fq:end-ahead)'    cont_RV_LM_w((22-4)*fq:end-ahead)'    cont_RV_LM_m(1:end-ahead)'];
                   
Regressor_BPV= [BPV_X_d(22*fq:end-ahead)'    BPV_X_w((22-4)*fq:end-ahead)'    BPV_X_m(1:end-ahead)'];

else
    error
    
end
%%

if strcmp(pred_freq,'daily')
    
  y=total_RV_d(22*fq+ahead:end);   % Because right now I try to predict "total_RV_d"

elseif strcmp(pred_freq,'weekly')

  y=total_RV_w((22-4)*fq+ahead:end);   % Because right now I try to predict "total_RV_w"

elseif strcmp(pred_freq,'monthly')

  y=total_RV_m(1*fq+ahead:end);   % Because right now I try to predict "total_RV_m"

else
    
    error

end
                        

lmd_plain=fitlm(Regressor_total_RV,y,'linear');

lmd_NN=fitlm(Regressor_cont_RV_NN,y,'linear');

lmd_LM=fitlm(Regressor_cont_RV_LM,y,'linear');

lmd_BPV=fitlm(Regressor_BPV,y,'linear');

%%
% Below are the regressors that I will use to forecast out-of-sample (so not for the
% estimation of coefficients).
% This is why I put ONLY the last observation.
% The history of observations is used above to estimate the coefficients.
% The LAST observation is used (outside of this function) to make the prediction.

if strcmp(incl_j_X,'yes') % Include the jump components as regressors
    
Regressor_total_RV_last= [total_RV_d(end)' total_RV_w(end)' total_RV_m(end)'];

Regressor_cont_RV_NN_last= [cont_RV_NN_d(end)'    cont_RV_NN_w(end)'   cont_RV_NN_m(end)' ...
                                 JV_NN_d(end)'         JV_NN_w(end)'        JV_NN_m(end)'];                
                            
Regressor_cont_RV_LM_last= [cont_RV_LM_d(end)'    cont_RV_LM_w(end)'    cont_RV_LM_m(end)' ...
                                 JV_LM_d(end)'         JV_LM_w(end)'         JV_LM_m(end)'];
                                                         
Regressor_BPV_last= [BPV_X_d(end)'    BPV_X_w(end)'    BPV_X_m(end)' ...
                    JV_BPV_d(end)'   JV_BPV_w(end)'   JV_BPV_m(end)'];
                
elseif strcmp(incl_j_X,'no') % Don't include the jump components

Regressor_total_RV_last= [total_RV_d(end)' total_RV_w(end)' total_RV_m(end)'];

Regressor_cont_RV_NN_last= [cont_RV_NN_d(end)'    cont_RV_NN_w(end)'   cont_RV_NN_m(end)'];               
                            
Regressor_cont_RV_LM_last= [cont_RV_LM_d(end)'    cont_RV_LM_w(end)'    cont_RV_LM_m(end)'];
                          
Regressor_BPV_last= [BPV_X_d(end)'    BPV_X_w(end)'    BPV_X_m(end)'];

else
    error
end
                           
end
