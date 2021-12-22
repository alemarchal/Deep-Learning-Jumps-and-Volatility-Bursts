%%%%%% Neural network that classifies a time series of returns into "jump"
%%%%%% or "no jump". In this file we do the TRAINING + TESTING
%% generate simulated data
close all
clear all
clc
%% (1) Parameters Monte-Carlo simulation

frequency = 2; %Frequency of the data (in minutes)
%Years = 2.5; % #of years
Years=0.5;
T = Years; % In years
NT = Years*248*6.5*60/frequency;% Number of time steps
%NT = Years * 249 * 386/frequency;
dt = T/NT; %dt = time interval of the data (in minutes)

%NSP = 70; %Number Simulations PER paramater value
NSP=2;

V0 = 100;    % Initial firm value
rf = 0.05;   % Risk-free rate (yearly)
q  = 0.0;    % Dividend
sigma = 0.15; % Diffusion coefficient
alpha=600.0;
%vol='sv_JD';
%vol='sv_JD';
vol='sv_JD';

price='2F';

sJc = 0.9;
c=0.025;
% c=0.06 at 15 min
% Jump parameters for the prices
lambdaJ = 100.0; % Jump intensity (average number of jumps PER YEAR). For 3 years it's lambdaJ*3
muJ     = 0.00; % Mean of the jump size distribution

% Jump parameters for the volatility
lambdaJ_vol = 65;
mu_vol = 0.18;



fc = 10^1; % scaling constant
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Generate training data %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


b=1;
e=NSP;

set_sJc = [0.6 0.8 1 1.5 2.2];

NS = NSP*length(set_sJc); % Total Number of Paths



V = zeros(NT,NS);

for i=1:length(set_sJc)

    sJc = set_sJc(i); 

[V_temp,True_jump_times_temp] = Jump_Diffusion_sim(V0,rf,q,sigma,lambdaJ,muJ,NT,NSP,dt,c,sJc,vol,alpha,lambdaJ_vol,mu_vol,price);

V(:,b:e)=V_temp;
True_jump_times(:,b:e)=True_jump_times_temp;

b=b+NSP; %beginning (temp variable)
e=e+NSP; %end (temp variable)
end


% Compute the log-returns
r_ln = log(V(2:end,:) ./ V(1:end-1,:));%Vector of log-returns for all the paths


% Compute the Bipower Variation for all the paths
K_window = ceil(dt^(-0.52));
Bipower_StD=nan(length(K_window+1:length(V(:,1))),NS);
for p=1:NS 
V_test=V(:,p);
    i=0;
    for ti=K_window+1:length(V_test) % Jump test time
    i=i+1;
    Bipower_StD(i,p) = BV(K_window,V_test,ti);
    end
end





r_ln_standard = r_ln(K_window:end,:)./Bipower_StD; % Divides each path by its corresponding standard deviation
% Truncate the first K_window returns because for them I can't compute the Bipower variation



% Here I classify the data in either (i) Jump or (ii) No-Jump

Jump_ind = zeros(NT,NS);
for i=1:length(True_jump_times(1,:)) % For each column (for each time series) I want a vector with 1 and 0 telling me if there is a jump
    %True_jump_times contains the JUMP TIME. For for building CATEGORIES
    %it's easier to replace this jump time with a ONE.
    index=0;
    index = find(True_jump_times(:,i)~=0);
    
Jump_ind(index,i) =  1;

end

XTemp= {0};
for i=1:NS
    
 
   XTemp(i)= { (r_ln_standard(:,i).*fc)' }; % Use the RETURNS of the stochastic process as training data
 
end

YTemp={0};
for j=1:NS

 % YTemp{j} =categorical(Jump_ind(:,j)',[0 1],{'no jump' 'jump'});  % Use the PATHS of the stochastic process as training data

  YTemp{j} =categorical(Jump_ind(2+K_window-1:end,j)',[0 1],{'no jump' 'jump'});  % Use the RETURNS of the stochastic process as training data

end



XTrain = XTemp';
YTrain = YTemp';
%% Plot prices for the introduction of the slides
% 
% frequency = 2; %Frequency of the data (in minutes)
% %Years = 2.5; % #of years
% Years=1/80;
% T = Years; % In years
% NT_plot = Years*256*6.5*60/frequency;% Number of time steps ONLY FOR THIS PLOT
% dt_plot = T/NT_plot; %dt = time interval of the data (in minutes)
% 
% 
% NS=1;
% 
% lambdaJ = 1500.0; %
% sJc = 0.65;
% lambdaJ_vol = 650;
% 
% rng(9) %Freeze the random generator
% [V,True_jump_times] = Jump_Diffusion_sim(V0,rf,q,sigma,lambdaJ,muJ,NT_plot,NS,dt_plot,c,sJc,vol,alpha,lambdaJ_vol,mu_vol,price);
% 
% index = find(True_jump_times~=0);
% 
% V_noJump=V;
% V_noJump(index)=nan;
% 
% 
% figure
% plot(V_noJump,'-o','MarkerEdgeColor','red','MarkerIndices',index)
% xlabel("Time")
% ylabel("Price")
% ax = gca;
% ax.FontSize = 12;
% 
% 
% figure
% plot(V,'o','Markersize',3,'MarkerFaceColor', 'b')
% xlabel("Time")
% ylabel("Price")
% ax = gca;
% ax.FontSize = 12;
%  
% % r_ln = log(V(2:end,:) ./ V(1:end-1,:));%Vector of log-returns for all the paths
% % figure
% % plot(r_ln)


%% Plot returns for the image with 4 boxes summarizing our entire methodology


% frequency = 2; %Frequency of the data (in minutes)
% %Years = 2.5; % #of years
% Years=1/80;
% T = Years; % In years
% NT_plot = Years*256*6.5*60/frequency;% Number of time steps ONLY FOR THIS PLOT
% dt_plot = T/NT_plot; %dt = time interval of the data (in minutes)
% 
% 
% NS=1;
% 
% lambdaJ = 2500.0; %
% sJc = 0.5;
% lambdaJ_vol = 650;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%% For the LABEL graph %%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% rng(71) %Freeze the random generator
% 
% [V,True_jump_times] = Jump_Diffusion_sim(V0,rf,q,sigma,lambdaJ,muJ,NT_plot,NS,dt_plot,c,sJc,vol,alpha,lambdaJ_vol,mu_vol,price);
% 
% 
% r_ln = log(V(2:end,:) ./ V(1:end-1,:));%Vector of log-returns for all the paths
% 
% 
% index_for_price = find(True_jump_times~=0);
% 
% index_for_returns = index_for_price-1;
% 
% size_marker = 10;
% 
% figure
% plot(1:length(r_ln),r_ln,'o','Markersize',size_marker,'MarkerFaceColor', 'b')
% xlabel("Time")
% %ylabel("Price")
% ax = gca;
% ax.FontSize = 12;
% 
% hold on
% 
% plot(index_for_returns,r_ln(index_for_returns),'o','Markersize',size_marker,'MarkerFaceColor', 'green')
% xlabel("Time")
% %ylabel("Price")
% ax = gca;
% ax.FontSize = 12;
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%% For the CLASSIFY graph %%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% rng(1) %Freeze the random generator
% 
% [V,True_jump_times] = Jump_Diffusion_sim(V0,rf,q,sigma,lambdaJ,muJ,NT_plot,NS,dt_plot,c,sJc,vol,alpha,lambdaJ_vol,mu_vol,price);
% 
% 
% r_ln = log(V(2:end,:) ./ V(1:end-1,:));%Vector of log-returns for all the paths
% 
% figure
% plot(1:length(r_ln),r_ln,'o','Markersize',size_marker,'MarkerFaceColor', 'b')
% xlabel("Time")
% %ylabel("Price")
% ax = gca;
% ax.FontSize = 12;

%% Plot training data

which=1; %Which path I want to plot

X = XTrain{which};
classes = categories(YTrain{which});

figure
for j = 1:numel(classes)
    label = classes(j);
    idx = find(YTrain{which} == label);
    hold on
    plot(idx,X(idx),'o','MarkerSize',2)
    %plot(X)
end
hold off

xlabel("Time Step")
ylabel("returns")
legend(classes,'Location','northwest')


%% Train the neural network

numFeatures = 1; % 1 dimension
numHiddenUnits = 100;
numClasses = 2; % 2 different categories (jump or no-jump)

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

whichAx = [false, true]; % [bottom, top]

options = trainingOptions('adam', ...
    'MaxEpochs',300, ...
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'Plots','training-progress', ...
    'OutputFcn', @(x)makeLogVertAx(x,whichAx));

    
net = trainNetwork(XTemp,YTrain,layers,options);


%% Generate test data
% Here I simulate data on which I will try to detect the jumps (the NN has
% not been trained on this data set so it is new <-> out of sample)

%save('TrainedNeuralNetwork','net')

% load net % Load the network (that was already trained)

trials=2000;
alpha=600.0;
%%%% Modify the parameters for a true out-of-sample classification %%%%
sigma = 0.1; % Diffusion coefficient
% Jump parameters
lambdaJ = 100.0; % Jump intensity (average number of jumps PER YEAR). For 3 years it's lambdaJ*3
lambdaJ_vol = 65; % Intensity of the jumps inside the volatility
sJc = 1.1;


%vol='real';
%vol='sv_PJ';

price='1F';

% alpha should be much bigger than lambdaJ_vol (like 10 times larger)

Ptg_detection_NN=nan(1,trials);
Ptg_spurious_NN=nan(1,trials);
Ptg_detection_LM=nan(1,trials);
Ptg_spurious_LM=nan(1,trials);

for t=1:trials
    
NS=1; % Here I test it on 1 path


[V_test,True_jump_times] = Jump_Diffusion_sim(V0,rf,q,sigma,lambdaJ,muJ,NT,NS,dt,c,sJc,vol,alpha,lambdaJ_vol,mu_vol,price);
%V_test is the vector of prices for path 1

Jump_ind_test = zeros(NT,NS);
for i=1:length(True_jump_times(1,:)) % For each column (for each time series) I want a vector with 1 and 0 telling me if there is a jump
    %True_jump_times contains the JUMP TIME. For for building CATEGORIES
    %it's easier to replace this jump time with a ONE. That's why I create Jump_ind_test.
    index=0;
    index = find(True_jump_times(:,i)~=0);
    
Jump_ind_test(index,i) =  1;
end



YTemptest={0};
for j=1:NS
  YTemptest{j}=categorical(Jump_ind_test(2+K_window-1:end,j)',[0 1],{'no jump' 'jump'});  % Use the RETURNS of the stochastic process as training data
% I shift by K_window-1 because I don't try to estimate the first K_window jumps
end
YTest = YTemptest{1,1}; % YTest is a vector containing the TRUE jumps (known because I simulate them)

  
r_ln_test = log(V_test(2:end) ./ V_test(1:end-1));%Vector of log-returns for the unique path


% Compute the Bipower Variation for all the paths
Bipower_StD=nan(length(K_window+1:length(V_test(:,1))),1);
i=0;
for ti=K_window+1:length(V_test) % Jump test time
    i=i+1;
    Bipower_StD(i) = BV(K_window,V_test,ti);
end



   
XTest = ( (r_ln_test(K_window:end,:)./Bipower_StD) .*fc)'; % Using the RETURNS as test data


%%%% This plots the raw data BEFORE the NN classifies them.
%figure
%plot(XTest,'o','MarkerSize',2)
%title('Raw data before classification')

YPred = classify(net,XTest); % YPred is a vector containing the jumps DETECTED by the NN


[Ptg_detection_NN_temp,Ptg_spurious_NN_temp,Ptg_detection_LM_temp,Ptg_spurious_LM_temp]=Accuracy_NN(YPred,YTest,V_test,NT,dt,True_jump_times(K_window:end));
 % I input True_jump_times(K_window:end) because the LM test doesn't detect
 % jumps on the first K_window returns. So if a jump really happened before K_window LM couldn't have detected it.
 % N.B: True_jump_times(K_window:end) is only used to assess the performance of LM

Ptg_detection_NN(t)=Ptg_detection_NN_temp;
Ptg_spurious_NN(t)=Ptg_spurious_NN_temp;
Ptg_detection_LM(t)=Ptg_detection_LM_temp;
Ptg_spurious_LM(t)=Ptg_spurious_LM_temp;

end

Avg_detection_NN=mean(Ptg_detection_NN)
Avg_spurious_NN=mean(Ptg_spurious_NN)
Avg_detection_LM=mean(Ptg_detection_LM)
Avg_spurious_LM=mean(Ptg_spurious_LM)




%%%%%%%%%%%%%% The part below is used to label the data in order to produce
%%%%%%%%%%%%%% a graph that says who detected what on SIMULATED DATA

% for the jumps predicted by the network
pos_jump_YPred = find(YPred=='jump')'+K_window-1;
ind_jump_NN = zeros(length(V_test)-1,1);
ind_jump_NN(pos_jump_YPred)=1;


% for the jumps predicted by LM
Stop_NT = length(V_test); %Length of the time series data
alpha_conf=0.01; % Confidence level
[Find_jump_times_LM,~,~,~]=LM_jump_test(alpha_conf,K_window,V_test,Stop_NT);
ind_jump_LM = zeros(length(V_test),1); 
ind_jump_LM(Find_jump_times_LM)=1;
ind_jump_LM=ind_jump_LM(2:end);

% for the TRUE jumps
temp_True_jump_times = find(True_jump_times~=0);
ind_jump_true = zeros(length(V_test),1); 
ind_jump_true(temp_True_jump_times)=1;
ind_jump_true=ind_jump_true(2:end);



diff_ind_jump = ind_jump_LM + 10*ind_jump_NN + 100*ind_jump_true;
YDiff=categorical(diff_ind_jump(K_window:end)',[0 1 10 11 100 101 110 111],{...
    'No jump & no detection both' ...
    'No jump & spurious LM' ...
    'No jump & spurious NN' ...
    'No jump & spurious both' ...
    'True jump & missed both' ...
    'True jump & detection LM' ...
    'True jump & detection NN' ...
    'True jump & detection both'});


%% Histogram for Confidence Intervals (CI)
bins=50;


% Figure size parameters
x0=100;
y0=100;
width=1200;
height=800;



figure
histogram(Ptg_detection_NN - Ptg_detection_LM, bins)
xlabel("% detection NN - % detection LM")
ax = gca;
ax.FontSize = 18;
set(gcf,'position',[x0,y0,width,height])
saveas(gcf,'hist_jumps_detection_NN_LM.png');


figure
histogram(Ptg_spurious_NN - Ptg_spurious_LM, bins)
xlabel("% spurious NN - % spurious LM")
ax = gca;
ax.FontSize = 18;
set(gcf,'position',[x0,y0,width,height])
saveas(gcf,'hist_jumps_spurious_NN_LM.png');




%% Visualize the results 
% This plot shows which jumps were detected by the L&M 2008 test AND the NN
% on SIMULATED DATA



X = XTest./fc;
classes = categories(YDiff);


exclude = ones(1,length(classes));
choice_color = {'#0072BD','r','#EDB120','#42B92F','k','r','#EDB120','#42B92F'};
choice_shape =  {'o','^'};
choice_size = [1.6 4];

figure
for j = 1:numel(classes)
    label = classes(j);

    
       idx = find(YDiff == label);
   if isempty(idx)
       exclude(j)=0;
   end
   
    hold on
   p= plot(idx,X(idx),choice_shape{1+double(j>4)},'MarkerSize',choice_size(1+double(j>1)),'color',choice_color{j});
   if j>1
   set(p, 'markerfacecolor', get(p, 'color')); % Use same color to fill in markers
   end
end
xlabel("Time")
ylabel("Standardized log-returns $\frac{r_t}{\hat{\sigma}_t}$","Interpreter","latex")
ax = gca;
ax.FontSize = 12;

pos_temp =find(exclude==1);
legend(classes(pos_temp),'Location','northwest')
% I use classes(pos_temp) to avoid the problem of "Warning: Ignoring extra legend entries" which shifts the legend and so messes up the graph.

%% Comparison with real data (Toy example; This is just a simple comparison. Not a real application of our methodology)


        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Using the real data %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%

load '../HF data/AIG_2min.mat'        % RENAME
data = AIG_2min;                   % RENAME



frequency = 2; %Frequency of the data (in minutes)
Years = 3; % #of years
T = Years; % In years
NT = Years*248*6.5*60/frequency;% Number of time steps
%NT = Years * 249 * 386/frequency;
dt = T/NT; %dt = time interval of the data (in minutes)

% Trading is from 9.30 until 16.00.
% So we have 195 observations per day (at 2 min frequency).

V_test=0;
V_test = data(1:end); % Using real data
%V_test = data(1:40000); % Using real data

index_nan = find(isnan(V_test)==1);
index_zero = find(V_test==0);


V_test=V_test(V_test~=0); %Get rid of the 0 to compute the log-returns


V_test(isnan(V_test)) = [] ; % Remove tha NaN


% Compute the Bipower Variation for the path
K_window = ceil(dt^(-0.5));
Bipower_StD=nan(length(K_window+1:length(V_test(:,1))),1);
i=0;
for ti=K_window+1:length(V_test) % Jump test time
    i=i+1;
    Bipower_StD(i) = BV(K_window,V_test,ti);
end

r_ln_test = log(V_test(2:end) ./ V_test(1:end-1));%Vector of log-returns for the unique path


XTest = ( (r_ln_test(K_window:end,:)./Bipower_StD) .*fc)';
YPred = classify(net,XTest);
pos_jump_YPred = find(YPred=='jump')'+K_window-1;

Stop_NT = length(V_test); %Length of the time series data
alpha_conf=0.01; % Confidence level

[Find_jump_times_LM,~,~,~]=LM_jump_test(alpha_conf,K_window,V_test,Stop_NT);


qqq = ismember(pos_jump_YPred,Find_jump_times_LM-1);

sum(qqq) % Intersection of the number of jumps detected by both tests

ind_jump_LM = zeros(length(V_test),1); 
ind_jump_LM(Find_jump_times_LM)=1;
ind_jump_LM=ind_jump_LM(2:end);

ind_jump_NN = zeros(length(V_test)-1,1);
ind_jump_NN(pos_jump_YPred)=1; % This is ALL the jumps classified by NN

%YLM=categorical(ind_jump_LM',[0 1],{'no jump' 'jump'}); ?useless?


diff_ind_jump = ind_jump_LM + 10*ind_jump_NN;
YDiff=categorical(diff_ind_jump(K_window:end)',[0 1 10 11],{'No jump both' 'Jump LM' 'Jump NN' 'Jump both'});

%% Visualize the results 
% This plot shows which jumps were detected by the L&M 2008 test AND the NN
% on REAL DATA.


X = XTest./fc;
classes = categories(YDiff);


exclude = ones(1,length(classes));
choice_color = {'#0072BD','r','#EDB120','#42B92F'};
choice_shape =  {'o','*'};
choice_size = [1.6 4.2];


figure
for j = 1:numel(classes)
    label = classes(j);
    
    idx = find(YDiff == label);
   
    if isempty(idx)
       exclude(j)=0;
   end
    
   hold on
   p= plot(idx,X(idx),choice_shape{1+double(j>4)},'MarkerSize',choice_size(1+double(j>1)),'color',choice_color{j});
   if j>1
   set(p, 'markerfacecolor', get(p, 'color')); % Use same color to fill in markers
   end
end
xlabel("Time")
ylabel("Standardized log-returns $\frac{r_t}{\hat{\sigma}_t}$","Interpreter","latex")
ax = gca;
ax.FontSize = 20;

pos_temp =find(exclude==1);
legend(classes(pos_temp),'Location','northwest')
% I use classes(pos_temp) to avoid the problem of "Warning: Ignoring extra legend entries" which shifts the legend and so messes up the graph.


%% Robustness check
% Here I simply compute the Realized Volatility (RV) of the real stock
% returns in order to use them later for the testing part (as a robustness
% check).


index_temp= YDiff=='Jump NN' | YDiff=='Jump both';

return_without_jump=X'.*Bipower_StD;
return_without_jump(index_temp) = [] ; % Remove the jumps

RV_window=10;
spotvar_estimate = RV(return_without_jump,RV_window)/(RV_window);

StD_TICKER = sqrt( spotvar_estimate );

%% Event study (Isolate a jump where the NN disagrees with LM and link it to a news)
% I do this event study for AIG


r_shifted=[r_ln_test(K_window:end)' ; 1:length(r_ln_test(K_window:end)) ]; % The 2nd row is the index of the return
V_shifted=V_test(K_window:end);
%
r_jump_NN_index = YDiff=='Jump NN'; % This is the jumps classified ONLY by NN (and not LM).
r_jump_NN_only = r_shifted(:,r_jump_NN_index);
%
r_jump_LM_index = YDiff=='Jump LM'; % This is the jumps classified ONLY by LM (and not NN).
r_jump_LM_only = r_shifted(:,r_jump_LM_index);
%
r_jump_both_index = YDiff=='Jump both'; % This is the jumps classified by both NN & LM.
r_jump_both = r_shifted(:,r_jump_both_index);



find(abs(r_jump_NN_only(1,:))== max(abs(r_jump_NN_only(1,:)))) % Find the largest 'jump NN' in absolute value

% For AIG this is the number 30 in r_jump_NN_only.
% The log return is [0.121771872217616]
% The index for this jump is [130461] in the r_shifted vector.
r_shifted(1,130461)


log( V_shifted(130461 +1)/V_shifted(130461) )

(130461+K_window+3)/193 %+3 because there are 3 NaN before. And + K_window because I want to find the position of this return in the full vector of prices (where I didn't remove K_window)
% Divide by 193 because I have 193 observations per day.
% So the jump will be +- at row 677 in the 'AIG_Y2006_2008M1_12_2min.mat'
% file.

load '../HF data/AIG_Y2006_2008M1_12_2min.mat' %This is the full file with the trading dates (original .mat file)

M1 = CLEANDATA_2min(:,:,1); %Matrix contening the TRADING DATES
M2 = CLEANDATA_2min(:,:,2); %Matrix with the PRICES

M2(678,22)
M2(678,23)
% Those give me the position in the matrix M1 for the dates


t1 = datetime(M1(678,22),'ConvertFrom','datenum') %date of the price before the jump
t2 = datetime(M1(678,23),'ConvertFrom','datenum') %date of the price after the jump

%%%%%%%%%%%%%%
%%%% Plot %%%%
%%%%%%%%%%%%%%
shift_m=21-8;
shift_p=193-120;

y=V_shifted(130461-shift_m:130461+shift_p);

b_j=14; % price point before the jump


startDate = datenum('19-09-2008 9:51:00');
endDate = datenum('19-09-2008 12:43:00');
xaxis = linspace(startDate,endDate,length(y));

sc=10^2; % Size Cercles. The size for scatter is the squared size of the plot (if I want same size dots)

figure
scatter(xaxis,y,sc,'filled','MarkerFaceColor','blue','MarkerEdgeColor','blue')
hold on
scatter(xaxis(b_j),y(b_j),sc,'MarkerFaceColor','green','MarkerEdgeColor','green') % Before I used #EDB120 color
hold on
scatter(xaxis(b_j+1),y(b_j+1),sc,'MarkerFaceColor','green','MarkerEdgeColor','green')
%xlabel('time (hour)')
ax = gca;
ax.XTick = [xaxis(1) xaxis(b_j) xaxis(end)]; %Display only the dates for these points on the x-axis
datetick(gca,'x','HH:MM','keepticks','keeplimits')
ylabel('AIG price')
xlabel('19 September 2008')
ax.FontSize = 35;

b = num2str(y); c = cellstr(b);
dx = 0.0; dy = 0.0; % displacement so the text does not overlay the data points
text([xaxis(b_j)+0.001 xaxis(b_j+1)-0.002]+dx, [y(b_j) (y(b_j+1)+0.045)]+dy, [c(b_j) c(b_j+1)],'FontSize',20);




