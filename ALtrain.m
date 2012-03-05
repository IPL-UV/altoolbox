function ALtrain(trnSet, modelParameters, num_of_classes, modelname, rundir)

% function ALtrain(trnSet, modelParameters, num_of_classes, modelname, rundir)
%
%  trnSet: training set
%  costFin: SVM C parameter
%  stdzFin: SVM RBF Gaussian kernel sigma parameter
%  num_of_classes: number of classes
%  modelname: filename where the model will be saved to
%  rundir: directory where 'multisvm' will run
%
% See also AL, ALtoolbox

enteteSTrnSet = [size(trnSet,1) size(trnSet,2)];
dlmwrite([rundir '/TrnSetShuffled.txt'], enteteSTrnSet, 'delimiter', ' ', 'precision', 8);
dlmwrite([rundir '/TrnSetShuffled.txt'], trnSet, 'delimiter', ' ', '-append');

% Entranement
cmdtrn = sprintf('./multisvm %s/TrnSetShuffled.txt %s %d -c %f -std %f -dir %s', ...
    rundir, modelname, num_of_classes, modelParameters.costFin, modelParameters.stdzFin * sqrt(2), rundir);

[status,result] = system(cmdtrn);
if status ~= 0
    disp(result)
    error('ALToolbox:AL:multisvm','Error running multisvm')
end
