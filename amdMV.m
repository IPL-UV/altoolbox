function selected_points_index = ...
    amdMV(trnSet, cndSet, num_of_classes, viewsVector, sample2add, svmDir, nfolds, sigmas, Cs)

% function selected_points_index = ...
%    amdMV(trnSet, cndSet, num_of_classes, viewsVector, sample2add, svmDir, nfolds, sigmas, Cs)
% Inputs:
%       - trnSet: Labled Data
%       - cndSet: Unlabeled data used for AL
%       - num_of_classes: Number of classes
%       - viewsVector: ex:[1 1 1 2 2 3 3 2 3] means 9 features with 3 views
%       - sample2add: batch size
%       - svmDir: SVM classifiers path for MV SVMs
%       - nfolds, sigmas, Cs: SVM hyperparameters tunning
% Outputs:
%       - selected_points_index: selected points to be added to the training
%                               data per step
%
% - by Hsiuhan Lexie Yang (2012.10)
%   We also thank Wei Di who contributes to the code skeleton.

% prep
no_view = length(unique(viewsVector));
viewIdx = unique(viewsVector);
USampleNo = size(cndSet, 1);

modelname = sprintf('%s/modelTestBoot_Schohn', svmDir);
options.model = 'SVM';

predictions_EV_U = zeros(USampleNo,no_view);
for i_view = 1:no_view
    subband_index = find(viewsVector == viewIdx(i_view));
    trnSet_view = [trnSet(:,subband_index) trnSet(:,end)];
    cndSet_view = [cndSet(:,subband_index) cndSet(:,end)];
    % start train SVM
    % tunning parameters
    modelParameters = GridSearch_Train_CV(trnSet_view, num_of_classes, sigmas, Cs, nfolds, svmDir);
    ALtrain(trnSet_view, modelParameters, num_of_classes, modelname, svmDir);
    % predictions of cndSet on trained SVM
    predictions_EV_U(:,i_view) = ALpredict(options.model, trnSet_view, cndSet_view, modelname, num_of_classes, svmDir);
end

% -- Disagreement measure based on predictions of each view
disagreement = zeros(USampleNo,1);
for i_instance = 1:USampleNo
    disagreement(i_instance) = length(unique(predictions_EV_U(i_instance,:)));
end
d_max = max(disagreement);  % Adaptive disagreement threshold
contention_points_index = find(disagreement == d_max);
no_contention_points = length(contention_points_index);

% -- Choose samples which meet maximum disagreement
if no_contention_points > sample2add
    contention_points = cndSet(contention_points_index,:);
    % randomly select points from the contention pool, just pick up the top
    % ones will be too restricted.
    perm = randperm(size(contention_points,1));
    temp_selected_index = perm(1:sample2add);
    selected_points_index = contention_points_index(temp_selected_index);
else
    selected_points_index = contention_points_index;
end
