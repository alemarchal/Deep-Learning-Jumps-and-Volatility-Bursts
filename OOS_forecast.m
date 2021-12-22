function [MSE_total_RV,MSE_NN,MSE_LM,MSE_BPV    ,    R2_total_RV,R2_NN,R2_LM,R2_BPV ] = OOS_forecast(net,data,end_data,cutoff,dt,one_day,step_forecast,fc,pred_freq,incl_j_X)

% Out Of Sample (OOS) forecast of future Realized Volatility (RV)


V_all=0;
%V_all = data(1:end_data); % Using real data
V_all = data(1:end); % Using real data
V_all=V_all(V_all~=0); %Get rid of the 0 to compute the log-returns
V_all(isnan(V_all)) = [] ; % Remove tha NaN



 % Here I classify the whole series of returns into "jump" / "no jump" once
 % and for all. Then I feed gradually the information to make out-of-sample
 % forecasts. It's not trully out-of-sample because of the bidirectionnal
 % layer in the network (but it's much faster !)

% Compute the Bipower Variation for the path
K_window = ceil(dt^(-0.5));
Bipower_StD_all=nan(length(K_window+1:length(V_all(:,1))),1);
i=0;
for ti=K_window+1:length(V_all) % Jump test time
    i=i+1;
    Bipower_StD_all(i) = BV(K_window,V_all,ti);
end


V_shift_all=V_all(K_window:end);


r_ln_test = log(V_all(2:end) ./ V_all(1:end-1));%Vector of log-returns for the unique path

XTest_all = ( (r_ln_test(K_window:end,:)./Bipower_StD_all) .*fc)';
YPred = classify(net,XTest_all);
pos_jump_YPred = find(YPred=='jump')'+K_window-1;

Stop_NT = length(V_all); %Length of the time series data
alpha_conf=0.01; % Confidence level for the LM test

[Find_jump_times_LM,~,~,~]=LM_jump_test(alpha_conf,K_window,V_all,Stop_NT);


ind_jump_LM = zeros(length(V_all),1); 
ind_jump_LM(Find_jump_times_LM)=1;
ind_jump_LM=ind_jump_LM(2:end);
ind_jump_NN = zeros(length(V_all)-1,1);
ind_jump_NN(pos_jump_YPred)=1;

diff_ind_jump = ind_jump_LM + 10*ind_jump_NN;
YDiff_all=categorical(diff_ind_jump(K_window:end)',[0 1 10 11],{'No jump both' 'Jump LM' 'Jump NN' 'Jump both'});

%% Compute the continuous RV (I remove jumps according to the 2 different methods)

% Here I compute the regressors once and for all and then I will feed them
% gradually to the function that peforms the regression (it's way faster).


%fq = one_day; % choose this if I use RV at the highest frequency;
fq=1; % choose this if I sample RV using one observation per day


% ATTENTION: this code doesn't work for "fq = one_day;" and using RV
% computed at the highest frequency (there is a problem in the regression
% with shifting correctly the vectors). I should fix that !


index_temp_NN= YDiff_all=='Jump NN' | YDiff_all=='Jump both';
index_temp_LM= YDiff_all=='Jump LM' | YDiff_all=='Jump both';

X = XTest_all./fc;


return_without_jump_NN=X'.*Bipower_StD_all;

% prec_return = zeros(length(X),1);
% prec_return(find(index_temp_NN==1)-1)=1;
% temp=logical(prec_return);
% return_without_jump_NN(index_temp_NN) = return_without_jump_NN(temp);
return_without_jump_NN(index_temp_NN) = 0.0;%[] ; % Remove the jumps detected by the NEURAL NETWORK
% The problem about simply removing the jumps is that it shifts the time in
% the time series. This messes up the predictions.
 % What I do instead:
%if there was a jump, probably the previous return wasn't a jump. So I substitute the jump with the value of the previous return.
% Or I can just put 0 for the moment. TO BE CHANGED AFTER !!!!!

return_without_jump_LM=X'.*Bipower_StD_all;

% prec_return = zeros(length(X),1);
% prec_return(find(index_temp_LM==1)-1)=1;
% temp=logical(prec_return);
% return_without_jump_LM(index_temp_LM) = return_without_jump_LM(temp);
return_without_jump_LM(index_temp_LM) = 0.0;


return_with_jump=X'.*Bipower_StD_all;



%%%%%%%%%%%%%%%%%%%%%%%
%%% Total variation %%%
%%%%%%%%%%%%%%%%%%%%%%%
total_RV_full = sqrt(RV(return_with_jump,one_day)); % This gives the RV computed at the highest frequency possible
%total_RV_d=total_RV_full;
total_RV_d=total_RV_full(1:one_day:end); % This creates a vector with the level of RV every day (sampling at daily frequency)

total_RV_w = aggregate(total_RV_d,5*fq) ;
total_RV_m = aggregate(total_RV_d,22*fq);

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Bipower variation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Here it is not the Bipower variation used for the test of Lee & Mykland
% (the window will be different). Here it's the BPV used as a regressor in
% the regressions for prediction.

K_pred = one_day;%This is the difference with LM. Here I don't use "ceil(dt^(-0.5))".
BPV_X_full=nan(length(K_pred+1:length(V_shift_all(:,1))),1); %BPV_X is the Bipower variation used as a regressor.
i=0;
for ti=K_pred+1:length(V_shift_all) % Jump test time
    i=i+1;
    BPV_X_full(i) = sqrt((BV(K_pred,V_shift_all,ti)^2)*(K_pred-2)); % I do this because I want INTEGRATED BPV and not instantaneous (so I have do undo what LM do in their paper by multiplying by the window)
end

BPV_X_d=BPV_X_full(1:one_day:end)';
BPV_X_w = aggregate(BPV_X_d,5*fq);
BPV_X_m = aggregate(BPV_X_d,22*fq);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Continuous RV (we first remove the jumps then we compute RV) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cont_RV_NN_full = sqrt(RV(return_without_jump_NN,one_day));
cont_RV_NN_d=cont_RV_NN_full(1:one_day:end);

cont_RV_LM_full = sqrt(RV(return_without_jump_LM,one_day));
cont_RV_LM_d=cont_RV_LM_full(1:one_day:end);


cont_RV_NN_w = aggregate(cont_RV_NN_d,5*fq);
cont_RV_LM_w = aggregate(cont_RV_LM_d,5*fq);


cont_RV_NN_m = aggregate(cont_RV_NN_d,22*fq);
cont_RV_LM_m = aggregate(cont_RV_LM_d,22*fq);
%%%%%%%%%%%%%%%%%%%%%%
%%% Jump variation %%%
%%%%%%%%%%%%%%%%%%%%%%
JV_NN_full = sqrt( RV(return_with_jump,one_day) - RV(return_without_jump_NN,one_day) );
JV_NN_d=JV_NN_full(1:one_day:end);

JV_LM_full = sqrt( RV(return_with_jump,one_day) - RV(return_without_jump_LM,one_day) );
JV_LM_d=JV_LM_full(1:one_day:end);

JV_BPV_full = max( sqrt(RV(return_with_jump,one_day) - (BPV_X_full').^2 ), 0 );
JV_BPV_d = JV_BPV_full(1:one_day:end);


JV_NN_w = aggregate(JV_NN_d,5*fq);
JV_LM_w = aggregate(JV_LM_d,5*fq);
JV_BPV_w = aggregate(JV_BPV_d,5*fq);

JV_NN_m = aggregate(JV_NN_d,22*fq);
JV_LM_m = aggregate(JV_LM_d,22*fq);
JV_BPV_m = aggregate(JV_BPV_d,22*fq);



%% OUT-OF-SAMPLE forecast

% Here I cannot simply use the vector of detected jumps already classified
% by the NN and then use the point at time t to forecast the future.
% Because the NN uses a BIDIRECTIONAL layer to classify jumps so to
% classify the point at time t will use future information.

% Instead I should feed the data as a loop and re-classify the jumps at
% each iteration & I also want to re-estimate the coefficients of the
% predictive regression everytime a new piece of information arrives.


total_RV_d_history=0;
total_RV_w_history=0;
total_RV_m_history=0;

yHat_total_RV_temp=0;
yHat_NN_temp=0;
yHat_LM_temp=0;
yHat_BPV_temp=0;



i=0;
%for t=1:one_day:length(V_all)-cutoff-one_day
%for t=1:one_day:one_day*160
for t=1:1:length(total_RV_m)-cutoff%-max(one_day,220) % Since I sample RV at the daily frequency, I add one day worth of information at every iteration.
i=i+1;
    


total_RV_d_tc=total_RV_d(1:22*fq+cutoff+t-1); %TC stand for time consistent
total_RV_w_tc=total_RV_w(1:18*fq+cutoff+t-1);
total_RV_m_tc=total_RV_m(1:1+cutoff+t-1);

cont_RV_NN_d_tc=cont_RV_NN_d(1:22*fq+cutoff+t-1);
cont_RV_NN_w_tc=cont_RV_NN_w(1:18*fq+cutoff+t-1);
cont_RV_NN_m_tc=cont_RV_NN_m(1:1+cutoff+t-1);
JV_NN_d_tc=JV_NN_d(1:22*fq+cutoff+t-1);
JV_NN_w_tc=JV_NN_w(1:18*fq+cutoff+t-1);
JV_NN_m_tc=JV_NN_m(1:1+cutoff+t-1);

cont_RV_LM_d_tc=cont_RV_LM_d(1:22*fq+cutoff+t-1);
cont_RV_LM_w_tc=cont_RV_LM_w(1:18*fq+cutoff+t-1);
cont_RV_LM_m_tc=cont_RV_LM_m(1:1+cutoff+t-1);
JV_LM_d_tc=JV_LM_d(1:22*fq+cutoff+t-1);
JV_LM_w_tc=JV_LM_w(1:18*fq+cutoff+t-1);
JV_LM_m_tc=JV_LM_m(1:1+cutoff+t-1);

BPV_X_d_tc=BPV_X_d(1:22*fq+cutoff+t-1);
BPV_X_w_tc=BPV_X_w(1:18*fq+cutoff+t-1);
BPV_X_m_tc=BPV_X_m(1:1+cutoff+t-1);
JV_BPV_d_tc=JV_BPV_d(1:22*fq+cutoff+t-1);
JV_BPV_w_tc=JV_BPV_w(1:18*fq+cutoff+t-1);
JV_BPV_m_tc=JV_BPV_m(1:1+cutoff+t-1);


[lmd_plain , X_total_RV , lmd_NN , X_NN , lmd_LM , X_LM, lmd_BPV , X_BPV] = LR_jumps(fq,step_forecast,...
    pred_freq,total_RV_d_tc, total_RV_w_tc,total_RV_m_tc,cont_RV_NN_d_tc, cont_RV_NN_w_tc,cont_RV_NN_m_tc,...
    JV_NN_d_tc,JV_NN_w_tc,JV_NN_m_tc,cont_RV_LM_d_tc,cont_RV_LM_w_tc,cont_RV_LM_m_tc,JV_LM_d_tc,JV_LM_w_tc,...
    JV_LM_m_tc,BPV_X_d_tc,BPV_X_w_tc,BPV_X_m_tc,JV_BPV_d_tc,JV_BPV_w_tc,JV_BPV_m_tc,incl_j_X);% Re-estimate the betas of the regression everytime a new information arrives



total_RV_d_history(i)= X_total_RV(1); 
total_RV_w_history(i)= X_total_RV(2); 
total_RV_m_history(i)= X_total_RV(3); 

% Make a prediction (out-of-sample)
betaHat_total_RV = lmd_plain.Coefficients.Estimate;
X_total_RV_new = [ones(1,1) X_total_RV];
yHat_total_RV_temp(i) =  X_total_RV_new * betaHat_total_RV;


betaHat_NN = lmd_NN.Coefficients.Estimate;
X_NN_new = [ones(1,1) X_NN];
yHat_NN_temp(i) =  X_NN_new * betaHat_NN;


betaHat_LM = lmd_LM.Coefficients.Estimate;
X_LM_new = [ones(1,1) X_LM];
yHat_LM_temp(i) =  X_LM_new * betaHat_LM;



betaHat_BPV = lmd_BPV.Coefficients.Estimate;
X_BPV_new = [ones(1,1) X_BPV];
yHat_BPV_temp(i) =  X_BPV_new * betaHat_BPV;

end


yHat_total_RV = yHat_total_RV_temp(1:end-step_forecast); % I throw away the last "step_forecast" predictions because I can't assess their accuracy
yHat_NN = yHat_NN_temp(1:end-step_forecast);
yHat_LM = yHat_LM_temp(1:end-step_forecast);
yHat_BPV = yHat_BPV_temp(1:end-step_forecast);


if strcmp(pred_freq,'daily')
    
    total_RV_history = total_RV_d_history; % Because right now I try to predict "total_RV_d"

elseif strcmp(pred_freq,'weekly')

    total_RV_history = total_RV_w_history; % Because right now I try to predict "total_RV_w"

elseif strcmp(pred_freq,'monthly')

    total_RV_history = total_RV_m_history; % Because right now I try to predict "total_RV_m"

else
    
    error

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Performance metrics %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MSE_total_RV = sqrt(sum( (yHat_total_RV - total_RV_history(1+step_forecast:end)).^2 )/length(yHat_total_RV)) *100; % Mean Square Error
MSE_NN = sqrt(sum( (yHat_NN - total_RV_history(1+step_forecast:end)).^2 )/length(yHat_NN)) *100;
MSE_LM = sqrt(sum( (yHat_LM - total_RV_history(1+step_forecast:end)).^2 )/length(yHat_LM)) *100;
MSE_BPV = sqrt(sum( (yHat_BPV - total_RV_history(1+step_forecast:end)).^2 )/length(yHat_BPV)) *100;

R2_total_RV = lmd_plain.Rsquared.Adjusted *100;
R2_NN = lmd_NN.Rsquared.Adjusted *100;
R2_LM = lmd_LM.Rsquared.Adjusted *100;
R2_BPV = lmd_BPV.Rsquared.Adjusted *100;

% MAE_total_RV = (sum( abs(yHat_total_RV - total_RV_history(1+step_forecast:end)) )/length(yHat_total_RV)) *100 % Mean Square Error
% MAE_NN = (sum( abs(yHat_NN - total_RV_history(1+step_forecast:end)) )/length(yHat_NN)) *100
% MAE_LM = (sum( abs(yHat_LM - total_RV_history(1+step_forecast:end)) )/length(yHat_LM)) *100




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% PLOTS of the 3 forecasting methods %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xaxis_total_RV = [1:length(total_RV_history)];
xaxis_yhat = [length(total_RV_history)-length(yHat_total_RV)+1:length(total_RV_history)];


% figure
% plot(xaxis_total_RV,total_RV_history,xaxis_yhat,yHat_total_RV,xaxis_yhat,yHat_NN,xaxis_yhat,yHat_LM)
% legend('RV monthly','Prediction total RV','Prediction NN','Prediction LM')
% 
% 
% 
% figure
% plot(xaxis_yhat,(yHat_total_RV-total_RV_history(1+step_forecast:end)).^2,xaxis_yhat,(yHat_NN-total_RV_history(1+step_forecast:end)).^2,xaxis_yhat,(yHat_LM-total_RV_history(1+step_forecast:end)).^2)
% legend('Error total RV','Error NN','Error LM')



%figure
%plot(xaxis_total_RV,MSE_total_RV,xaxis_total_RV,MSE_NN,xaxis_total_RV,MSE_LM)
%legend('MSE total RV','MSE NN','MSE LM')

end