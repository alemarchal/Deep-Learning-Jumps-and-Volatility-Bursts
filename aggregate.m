function [aggregate_object] = aggregate(object,nbr_days)



j=0;
for i=nbr_days:length(object)
   
   j=j+1; 
    
    aggregate_object(j) =  sum(object(i-nbr_days+1:i))/nbr_days ;
    
end


end