function [labels distances] = ALpredict(model, trnSet, valSet, modelname, num_of_classes, rundir)

% function [labels distances] = ALpredict(model, trnSet, valSet, modelname, rundir)
%
% Returns LDA / SVM predictions
%
%  model:     model used in predictions (see AL)
%  trnSet:    training set (needed for LDA_*)
%  valSet:    validation set
%  modelname: SVM model to use
%  num_of_classes: number of classes the model was trained
%  rundir:    this where 'multisvm' should be run
%
%  labels: predictions
%  distances: for SVM, the distances resulting of the one-against-all multiclass strategy
%
% See also AL, ALtoolbox

% Blocksize for predictions
blocksize = 1e4;

cmdval = sprintf('multisvm --val %s %s/tst.txt -dir %s', modelname, rundir, rundir);
if ~ispc, cmdval = ['./' cmdval]; end

labels = zeros(size(valSet,1),1);

% classes = unique(valSet(:,end)); % can fail, maybe using unique(trnSet(:,end))
% distances = zeros(size(valSet,1),length(classes));
% For now, use provided num_of_classes
distances = zeros(size(valSet,1), num_of_classes);

for be = 1:blocksize:size(valSet,1)
    % Indices
    idx = be:min(be+blocksize-1, size(valSet,1));
    % Predictions
    if strcmpi(model, 'LDA')
        labels(idx,:) = classify(valSet(idx,1:end-1),trnSet(:,1:end-1),trnSet(:,end));
    else
        % Test set
        enteteTstSet = [length(idx) size(valSet,2)];
        dlmwrite([rundir '/tst.txt'], enteteTstSet, 'delimiter', ' ', 'precision', 8);
        dlmwrite([rundir '/tst.txt'], valSet(idx,:), 'delimiter', ' ', '-append');
        
        [status,result] = system(cmdval);
        if status ~= 0
            disp(result)
            error('ALToolbox:AL:multisvm','Error running multisvm')
        end
        
        preds = textread([rundir '/predictions.dat']);
        labels(idx,:) = preds(:,1);
        distances(idx,:) = preds(:,2:end);
    end
end
