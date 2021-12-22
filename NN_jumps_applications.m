%%%%%% This files applies our jump classification method (developed
%%%%%% somewhere else) to economic problems.

%%
close all
clear all
clc
%%
load TrainedNeuralNetwork
fc = 10^1; % scaling constant

frequency = 2; %Frequency of the data (in minutes)
Years = 3; % #of years
T = Years; % In years
NT = Years*248*6.5*60/frequency;% Number of time steps
%NT = Years * 249 * 386/frequency;
dt = T/NT; %dt = time interval of the data (in minutes)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Realized Vol prediction %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if pred_freq =
%                'daily'        ==> predict RV_d
%                'weekly'       ==> predict RV_w
%                'monthly'      ==> predict RV_m
%                'all'          ==> predict all the 3 frequencies



% Here I use ALL the time series data. This is the FULL list of tickers.
% tickers = ["AA", "AIG", "AXP", "BA", "C", "CAT", "DD", "DIA", "DIS", "GE", ...
%     "GM", "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", ...
%     "MMM", "MO", "MRK", "MSFT", "PFE", "PG", "PWI", "T", "UTX", "VZ", ...
%     "WMT", "XOM"];



step_forecast = 1; % the unit is a day
pred_freq='all'; % to PREDICT {RV_d ; RV_w ; RV_m} depending on the selected frequency.

one_day = 193;%; % x observations at 2 min during one day
end_data = 400*193;
%cutoff = 30000; %where do I cut the data sample to estimate the regression. The rest will be used for forecasting.
cutoff=250; %The unit is days


incl_j_X='yes'; % Include jump components as regressors when forecasting future volatility.

% Here I remove AIG & AXP & C & JPM & PWI   (i.e. remove the financial stocks)
tickers = ["AA", "BA", "CAT", "DD", "DIA", "DIS", "GE", ...
    "GM", "HD", "HON", "HPQ", "IBM", "INTC", "JNJ","KO", "MCD", ...
    "MMM", "MO", "MRK", "MSFT", "PFE", "PG", "T", "UTX", "VZ", ...
    "WMT", "XOM"];


% writetable(final_table,'output_table.xls')      %To ouput the table as an Excel file (which after can be converted to a LaTeX table)
%%








if strcmp(pred_freq,'daily')
    
    [Performance_table] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast,fc,pred_freq,incl_j_X)

elseif strcmp(pred_freq,'weekly')

    [Performance_table] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast,fc,pred_freq,incl_j_X)
    
elseif strcmp(pred_freq,'monthly')
    
    [Performance_table] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast,fc,pred_freq,incl_j_X)
    
    
elseif strcmp(pred_freq,'all')
    
     step_forecast_temp_d=1;%#days
     pred_freq_temp_d='daily';
    [PT_daily] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast_temp_d,fc,pred_freq_temp_d,incl_j_X);
    %PT = Performance Table
    
     step_forecast_temp_w=5;%#days
     pred_freq_temp_w='weekly';
    [PT_weekly] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast_temp_w,fc,pred_freq_temp_w,incl_j_X);
    
     step_forecast_temp_m=22;%#days
     pred_freq_temp_m='monthly';
    [PT_monthly] = format_ouput(tickers,net,end_data,cutoff,dt,one_day,step_forecast_temp_m,fc,pred_freq_temp_m,incl_j_X);
 
    
final_table = horzcat(PT_daily,PT_weekly(:,2:end),PT_monthly(:,2:end))


else
    error
    
end




















