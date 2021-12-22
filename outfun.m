function stop = outfun(x, optimValues, state)
% The goal of this function is to save the solution reached by the
% algorithm so far in case I stop it manually (using ctrl+c)


stop = false;

sol_temp = x;

% to plot the solution at each iteration
surf(1:13,1:33,x)
drawnow

save sol_temp

end