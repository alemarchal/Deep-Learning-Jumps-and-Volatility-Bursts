function [V,True_jump_times,True_jump_intens] = Jump_Diffusion_sim(V0,rf,q,sigma,lambdaJ,muJ,NT,NS,dt,c,sJc,vol,alpha,lambdaJ_vol,mu_vol,price)

% Merton Jump-Diffusion price by simulation

% S0 = spot price


% q  = dividend yield
% sigma = volatillity
% T = maturity
% lambdaJ = jump frequency
% muJ     = jump mean parameter
% sigmaJ  = jump volatility parameter
% NT = number of time steps
% NS = number of stock price paths

% vol = 'sv_JD' uses a finite activity jump + Brownian process for the volatility (i.e. jump-diffusion).
% vol = 'sv_Cauchy' uses a pure jump Cauchy process for the volatility (infinite variation & infinite activity)


% Conditional jump probabilities
proba_jump = lambdaJ*dt;
proba_no_jump = 1-proba_jump;

% Conditional jump probabilities for volatility
proba_jump_vol = lambdaJ_vol*dt;
proba_no_jump_vol = 1-proba_jump_vol;



% Conditional probability of a Poisson process
prob=[proba_no_jump,proba_jump];
prob=cumsum(prob);
value=[0,1];    % Nt will take value 0 with proba 1-lambdaJ*dt
                % and will take value 1 with proba lambdaJ*dt (proba_jump)

                
% Conditional probability of a Poisson process for jumps inside the
% volatility
prob_vol=[proba_no_jump_vol,proba_jump_vol];
prob_vol=cumsum(prob_vol);                
                
                
True_jump_times = zeros(NT,NS);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Added by Oksana

% Assume two jump states : 
% normal 0 with the param        and high 1 with param 
% prob muJ sigmaJ                prob*3 muJ sigmaJ*3

True_jump_state = zeros(NT,NS); 
prob_h=[1-proba_jump*2,proba_jump*2];
prob_h=cumsum(prob_h);

% End added by Oksana
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the stock price paths
V = zeros(NT,NS);
V(1,:) = V0;

Z1 = normrnd(0,1,NT,NS);
W1 = sqrt(dt)*Z1; %Wiener Process

SV = ones(NT,NS)*sigma;

vol_vol = sqrt(2*alpha*sigma)/4;


% Jump diffusion model for the volatility (finite activity Poisson process)
if strcmp(vol,'sv_JD')
Zvol =  normrnd(0,1,NT,NS);
Wvol = sqrt(dt)*Zvol;
    for s=1:NS
          for t=2:NT
               J_vol = 0;
              
              
               if lambdaJ_vol ~= 0 %(if lambdaJ is different from 0)
                        r_vol=rand;
                      
                        ind_vol=find(r_vol<=prob_vol,1,'first'); %Conditional probability of Nt for each time increment:
                        Nt_vol=value(ind_vol);  %Nt = 1 with probability lamdbaJ*dt
                             if Nt_vol > 0      %Nt = 0 with probability 1-lambdaJ*dt             
                                J_vol = J_vol + exprnd(mu_vol); 
                             end    
               end                             
            SV(t,s) = (SV(t-1,s) + alpha*(sigma - SV(t-1,s))*dt + sqrt(SV(t-1,s)) * vol_vol * Wvol(t-1,s) + J_vol);                         
          end
    end     
end
%%
% Pure jump CGMY process for the volatility (potentially with infinite variation & infinite activity)
if strcmp(vol,'sv_PJ')


    
% Parameters of the exponential transformation
rho= 0.8 ;
alpha_0= -0.70*1 ;
alpha_1= 1 ;
    
   
      
 % Parameters of the CGMY process   
 C= 8;
 G= 0.5;
 M= 0.5;
 Y= 1.4;
 T= dt*NT;
 t0=0;
 
 [CGMY]=CGMYprocess(C,G,M,Y,NS, NT ,T, t0 );
 
        
    
   SV(1,:) = sqrt( exp(alpha_0 + alpha_1 * -1.2) );
   f=ones(NT,NS)*-1.2;
     for s=1:NS
          for t=2:NT
%                f(t,s) = f(t-1,s) + rho * f(t-1,s)*dt + (CGMY(t,s)-CGMY(t-1,s));
%                SV(t,s) = sqrt( exp(alpha_0 + alpha_1 * f(t,s)) ) ;

                f(t,s) = f(t-1,s) + rho * (-1.2-f(t-1,s))*dt + (CGMY(t,s)-CGMY(t-1,s));
                SV(t,s) = sqrt( exp(alpha_0 + alpha_1 * f(t,s)) ) ;

          end 
     end             
end
%figure
%plot(SV)
%% Simulation of the second volatility coefficient (2nd factor)

Z2 = normrnd(0,1,NT,NS);
W2 = sqrt(dt)*Z2;

if strcmp(price,'2F')
sigma2=sigma;
SV2 = ones(NT,NS)*sigma2;
vol_vol2 = sqrt(2*alpha*sigma2)/4;



Zvol2 =  normrnd(0,1,NT,NS);
Wvol2 = sqrt(dt)*Zvol2;
    for s=1:NS
          for t=2:NT
               J_vol2 = 0;
              
              
               if lambdaJ_vol ~= 0 %(if lambdaJ is different from 0)
                        r_vol=rand;
                      
                        ind_vol=find(r_vol<=prob_vol,1,'first'); %Conditional probability of Nt for each time increment:
                        Nt_vol=value(ind_vol);  %Nt = 1 with probability lamdbaJ*dt
                             if Nt_vol > 0      %Nt = 0 with probability 1-lambdaJ*dt             
                                J_vol2 = J_vol2 + exprnd(mu_vol); 
                             end    
               end                             
            SV2(t,s) = SV2(t-1,s) + alpha*(sigma2 - SV2(t-1,s))*dt + sqrt(SV2(t-1,s)) * vol_vol2 * Wvol2(t-1,s) + J_vol2;                         
          end
    end     
   
              
            else
                SV2=zeros(NT,NS);
                
end

%%

if strcmp(vol,'real') % If vol=real then it's automatically a 2 factors (i.e. 2 Brownians inside the price process)

    
load StD_AA
load StD_AXP

    NS=1;
    
    SV=0;
    SV2=0;
    
   
    
    SV=StD_AXP(1:NT)'*0.2*10^3;
    SV2=StD_AA(1:NT)'*0.1*10^3;
%   SV2=zeros(NT,1);
 
    
end
%figure
%plot(SV)

%% Simulation of the PRICE process

for s=1:NS  % For each path
    JS = [0]; % Switch off if don't want autocorr in jumps
    for t=2:NT % For each time step
            J = 0; % What we add if no jump at this time step
            
            sigmaJ  = sJc *c* sqrt(SV(t-1,s)^2+SV2(t-1,s)^2); % StD of the jump size distribution
            
            % Added by Oksana
            state_innov = rand;
            state = 0.2*True_jump_state(t-1,s)+state_innov>=0.5;
            True_jump_state(t,s) = state;
            % End added by Oksana
                    if lambdaJ ~= 0 %(if lambdaJ is different from 0)
                        r=rand;
                        ind=find(r<=prob*(1-state)+prob_h*state,1,'first');    % State-dependent prob
                        % ind=find(r<=prob,1,'first'); %Conditional probability of Nt for each time increment:
                        Nt=value(ind);  %Nt = 1 with probability lamdbaJ*dt
                             if Nt > 0      %Nt = 0 with probability 1-lambdaJ*dt             
                               % J = J + normrnd(muJ,sigmaJ);                    % original   
                               % J = J + normrnd(muJ,sigmaJ+sigmaJ*2*state);   % Just high wol 
                               
                                J = (JS(end)*0.5)*state+normrnd(muJ,sigmaJ);  % Autoregressive
                                True_jump_times(t,s)=t;    %t;    
                                 JS=[JS,J]; % all the jumps of this path
                                if abs(J)<c* sqrt(SV(t-1,s)^2+SV2(t-1,s)^2) %In order to remove jumps which are very smalls (and so are not really jumps) 
                                    J=0;
                                    True_jump_times(t,s)=0;
                                end
                             end    
                    end
% Expected value of k, and drift term
nu = exp(muJ+0.5*sigmaJ^2) - 1;
drift = rf - q - lambdaJ*nu;
          
            
                
             V(t,s)=V(t-1,s)+ V(t-1,s)*(drift*dt + SV(t-1,s)*W1(t-1,s) + SV2(t-1,s)*W2(t-1,s) + J);   
       end
          
end



% A = True_jump_state*2+True_jump_times; % 0 and 2-no jump; 1- jump in low state; 3- jump in high state
% A(A==2) = 0; % now 0, 1 and 3
% True_jump_intens = A;

end