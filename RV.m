function [Realized_vol] = RV(returns,RV_window)
% Compute the Realized Variation of returns

r_squared = returns.^2;

j=0;
for i=RV_window:length(returns)
   
   j=j+1; 
    
    Realized_vol(j) =  sum(r_squared(i-RV_window+1:i)) ;
    
end


end