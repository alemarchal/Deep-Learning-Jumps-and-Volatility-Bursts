function [Find_jump_times,size_jump,RH0,L_i]=LM_jump_test(alpha,K_window,V_filtre,Stop_NT)

% Input arguments :
        % V_filtre : time series data to use
        % Stop_NT : length of the time series data we want to test


c = sqrt(2)/sqrt(pi); % constant

C_n = (sqrt(2*log(Stop_NT)))/c - ( log(4*pi)+log(log(Stop_NT)) )/(2*c*sqrt(2*log(Stop_NT))); % Corrected C_n (there was a mistake in the original paper by LM)s

S_n = 1/(c*sqrt(2*log(Stop_NT)));


%Rejection region
critical_value = -log(-log(1-alpha));

%Matrix pre-allocation

L_i =zeros(Stop_NT,1); 
jump_t_stat =zeros(Stop_NT,1);
RH0=zeros(Stop_NT,1);
size_jump = 0;



for ti=K_window+1:Stop_NT % Jump test time


    Bipower_StD = BV(K_window,V_filtre,ti);

    % Construction t-stat
    L_i(ti,1) =log(V_filtre(ti)/V_filtre(ti-1))/Bipower_StD; %T-stat
    jump_t_stat(ti,1) = (abs(L_i(ti,1))-C_n)/S_n; %Distribution of Maxima

    %Rejection rule
    RH0(ti) = jump_t_stat(ti,1) > critical_value;
    % =1 if we reject the null hypothesis of no jump.

                if RH0(ti)==1
                size_jump(ti,:) = log(V_filtre(ti)/V_filtre(ti-1));
                end
             
                
 end
 

size_jump=size_jump(size_jump~=0);
Find_jump_times = find(RH0==1);

end
