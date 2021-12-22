function [Performance_table] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast,fc,pred_freq,incl_j_X)




%% Perform the Out-of-Sample forecast

%for i=1:length(stickers) %Sequential loop
parfor i=1:length(tickers) %Parallel loop
stock_path=sprintf('HF data/%s_2min.mat',tickers(i));
stock_name=sprintf('%s_2min',tickers(i));
structured_file=load (stock_path);
data=structured_file.(stock_name);


[MSE_total_RV_cs(i),MSE_NN_cs(i),MSE_LM_cs(i),MSE_BPV_cs(i)    ,    R2_total_RV_cs(i),R2_NN_cs(i),R2_LM_cs(i),R2_BPV_cs(i) ]  = OOS_forecast(net,data,end_data,cutoff,dt,one_day,step_forecast,fc,pred_freq,incl_j_X);
end


%% Format the output results so that they look nice


MSE_total_RV = mean(MSE_total_RV_cs);
MSE_NN = mean(MSE_NN_cs);
MSE_LM  = mean(MSE_LM_cs);
MSE_BPV  = mean(MSE_BPV_cs);

R2_total_RV = mean(R2_total_RV_cs);
R2_NN = mean(R2_NN_cs);
R2_LM = mean(R2_LM_cs);
R2_BPV = mean(R2_BPV_cs);


% To display a nice table with the results
output_table = cell(4,3);

output_table{1,1} = 'PV'; % Plain Vanilla = total_RV
output_table{2,1} = 'BPV';
output_table{3,1} = 'LM';
output_table{4,1} = 'NN';


% The 2st column is for R^2
output_table{1,2} = R2_total_RV;
output_table{2,2} = R2_BPV;
output_table{3,2} = R2_LM;
output_table{4,2} = R2_NN;

% The 3st column is for MSE
output_table{1,3} = MSE_total_RV;
output_table{2,3} = MSE_BPV;
output_table{3,3} = MSE_LM;
output_table{4,3} = MSE_NN;

if strcmp(pred_freq,'daily')

    f='d';
    
elseif strcmp(pred_freq,'weekly')
   
    f='w';

elseif strcmp(pred_freq,'monthly')
    
    f='m';
    
else
    error
end
    



VarNames = {'Method',sprintf('R2_%s',f),sprintf('MSE_%s',f)};
Performance_table = table(output_table(:,1),output_table(:,2),output_table(:,3), 'VariableNames',VarNames);




%%
% figure
% plot(MSE_total_RV_cs)
% hold on
% plot(MSE_NN_cs)
% hold on
% plot(MSE_LM_cs)
% hold on
% plot(MSE_BPV_cs)

end