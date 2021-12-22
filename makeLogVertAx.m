function stop = makeLogVertAx(state, whichAx)
stop = false; % The function has to return a value.
% Only do this once, following the 1st iteration
if state.Iteration == 55
  % Get handles to "Training Progress" figures:
  hF  = findall(0,'type','figure','Tag','NNET_CNN_TRAININGPLOT_FIGURE');
  % Assume the latest figure (first result) is the one we want, and get its axes:
  hAx = findall(hF,'type','Axes');
  % Remove all irrelevant entries (identified by having an empty "Tag", R2018a)
  hAx = hAx(~cellfun(@isempty,{hAx.Tag}));
  hAx(whichAx).YScale = 'log';
end

end