function [accCurve, predictions, criterion, sampList, modelParameters] = ...
             AL(trnSet, cndSet, tstSet, num_of_classes, options)

% function [accCurve, predictions, criterion, sampList, modelParameters] = ...
%             ALcore(trnSet, cndSet, tstSet, num_of_classes, options)
%
% Active Learning toolbox: core function
% 
% Inputs:
%   - trnSet:         initial training points (ntr x dim + class)
%   - cndSet:         set of samples (candidate) for ranking (ncand x dim + class)
%   - num_of_classes: number of classes
%   - tstSet:         test set (ntest x dim + class)
%   - options:
%            .uncertainty:  'Random', 
%                           'EQB', (Tuia et al., Active learning methods for remote sensing image classification. IEEE TGRS, 2009)
%                           'MS', 
%                           'MCLU', (Demir et al., Batch mode active learning methods for the interactive classification of remote sensing images IEEE TGRS, 2011)
%                           'Multiview' (Di and Crawford, View Generation for Multiview Maximum Disagreement Based Active Learning for Hyperspectral Image Classification IEEE TGRS, 2012)
%
%            .diversity:    'None', 
%                           'ABD', (Demir et al., Batch mode active learning methods for the interactive classification of remote sensing images IEEE TGRS, 2011)
%                           'cSV', (Tuia et al., Active learning methods for remote sensing image classification. IEEE TGRS, 2009) 
%                           'Manifold' (Di and Crawford, Active Learning via Multi-view and Local Proximity Co-regularization for Hyperspectral Image Classification IEEE JSTSP, 2011)
%
%            .iterVector:  vector containing number of samples to add at each iteration
%            .model:       base classifier, 'LDA', 'SVM'
%            .pct:         for EQB, percentage of trainig pixels to make sub-trainig sets
%            .numModels:   for EQB, number of models to train
%            .normEQB:     - 0: without normalization [Tuia et al., TGRS, 2009]
%                          - 1: with normalization [Copa et al, SPIE, 2010]
%            .viewsVector: for Multiview, ex: [1 1 1 2 2 3] = 3 views with corresponding
%                          variables
%            .paramSearchIters: iterations where SVM hyperparameters must be adjusted.
%                               This is done using a grid search strategy, so it is slow
%                               and computationally demanding.
%   - Notes:
%     - MS and MCLU work only if 'SVM' chosen as classifier. If chosing 'LDA', we can use
%       posterior probabilities instead of decision functions to 'simulate' the criteria,
%       but it is at your own risk (ex: MCLU with posterior probabilities
%       corresponds to the Breaking ties criterion in Luo et al. JMLR 2005)
%     - Default options are: {'MS', 'None', 10, 'SVM', 0.6, 4, 1, [1 1 1 2 2 3], [1 10]}
%
% Outputs:
% 
%   - accCurve:        learning curve
%   - predictions:     classification predictions in test for each instance of options.iterVect
%   - criterion:       the uncertainty ranking of the candidate pixels, coming out of the
% 	                   uncertainty function
%   - sampList:        list of selected samples among the candidates
%   - modelParameters: RBF Gaussian kernel sigma and const C if using SVM
% 	
% by Devis Tuia, Jordi Mu\~noz-Mar\i', Hsiuhan Lexie Yang (2007-12)
%
% If using the toolbox, please cite:
%
% D. Tuia, M. Volpi, L. Copa, M. Kanevski, and J. Mu\~noz-Mar\'i. 
% A survey of active learning algorithms for supervised remote sensing image classification. 
% IEEE J. Sel. Topics Signal Proc., 5(3):606?617, 2011.
%
% See also ALToolbox

% Check parameters and set default options
if ~exist('options','var')
    options = struct();
end
if ~isfield(options, 'uncertainty')
    options.uncertainty = 'MS';
end
if ~isfield(options, 'diversity')
    options.diversity = 'None';
end
if ~isfield(options, 'iterVect')
    options.iterVect = 10;
end
if ~isfield(options, 'model')
    options.model = 'SVM';
end
if ~isfield(options, 'pct')
    options.pct = 0.6;
end
if ~isfield(options, 'numModels')
    options.numModels = 4;
end
if ~isfield(options, 'normEQB')
    options.normEQB = 1;
end
if ~isfield(options, 'viewsVector')
    options.viewsVector = [1 1 1 2 2 3];
end
if ~isfield(options, 'paramSearchIters')
    options.paramSearchIters = [1 10];
end

rundir = sprintf('./run_%s_%s', options.model, options.uncertainty);
modelname = '';
if ~isfield(options, 'modelParameters')
    options.modelParameters.stdzFin = [];
    options.modelParameters.costFin = [];
end

switch options.model
    case 'LDA'
    case 'SVM'
        if ~exist(rundir,'dir')
            mkdir(rundir)
        end
        modelname = sprintf('%s/modelTestBoot_Schohn', rundir);    
        % SVM training parameters
        nfolds = 3;
        sigmas = logspace(-2,1,5);
        Cs = logspace(0,2,5);
    otherwise
        error(['Unknown or unimplemented base model: ' options.model])
end

switch upper(options.uncertainty)
    case {'RANDOM', 'EQB'} 
         
    case {'MS', 'MCLU', 'MULTIVIEW'}
        if strcmpi(options.model, 'LDA')
            error([options.uncertainty ' uncertainty does not work with ' options.model])
        end
    otherwise
        error(['Unknown or unimplemented uncertainty: ' options.uncertainty])
end

switch upper(options.diversity)
    case {'NONE', 'ABD'} %, 'Manifold'} not yet implemented
    otherwise
        error(['Unknown or unimplemented diversity: ' options.diversity])
end
    
% Results
accCurve = zeros(numel(options.iterVect),2);
predictions = zeros(size(tstSet,1), numel(options.iterVect));

% Samples to add at each iteration
if length(options.iterVect) == 1
    criterion = cell(1);
    diffVect = options.iterVect;
else
    criterion = cell(numel(options.iterVect)-1,1);
    diffVect = diff(options.iterVect);
    
    % Reserve memory for sampList
    sampList = zeros(1,sum(diffVect));
    sampSize = 0;
    % This two allows us to fill sampList without using
    %   sampList = [sampList ; ... ]
end

% The remaining points from the candidates set
remPtsList = 1:size(cndSet,1);

% multisvm needs classes starting from 0 without 'gaps' between them.
% We build a translation table between the original classes and new classes satisfying
% this condition.

% Save original labels
trnLabels = trnSet(:,end);
cndLabels = cndSet(:,end);
tstLabels = tstSet(:,end);
% 1. Detect classes on original training set
trTable = unique(trnLabels);
% 2. Add classes on candidates set not present on the original training set
trTable = [trTable setdiff(trTable, cndLabels)];
% 3. The same for classes on test set
trTable = [trTable setdiff(trTable, tstLabels)];
% 4. Apply trTable
for i = 1:length(trTable)
    trnSet(trnLabels == trTable(i),end) = i-1;
    cndSet(cndLabels == trTable(i),end) = i-1;
    tstSet(tstLabels == trTable(i),end) = i-1;
end
clear trnLabels cndLabels tstLabels


% -----------------------------------------------------------------------------
% Main loop: iterVect contains the number of samples to add at each iteration
for ptsidx = 1:length(options.iterVect)
    
    % SVM training
    if strcmpi(options.model,'LDA')
        modelname = '';
    else
        % Search modelParameters when one of them is empty or when ptsidx is in paramSearchIters
        if ( isempty(options.modelParameters.stdzFin) || isempty(options.modelParameters.costFin) ) || ...
                ( ~isempty(find(ptsidx == options.paramSearchIters,1)) )
            options.modelParameters = GridSearch_Train_CV(trnSet, num_of_classes, sigmas, Cs, nfolds, rundir);
        end
        % Training
        ALtrain(trnSet, options.modelParameters, num_of_classes, modelname, rundir);
    end
    
    % Predictions on test set
    %disp('  Testing ...')
     
    % Predict in blocks to deal with large test sets
    predictions(:,ptsidx) = ...
        ALpredict(options.model, trnSet, tstSet, modelname, num_of_classes, rundir);
    
    % Classes in current training set
    classes = unique(trnSet(:,end));
    
    % compute test error and Kappa
    res = assessment(tstSet(:,end),predictions(:,ptsidx),'class');
    accCurve(ptsidx,1) = res.OA;
    accCurve(ptsidx,2) = res.Kappa;
    fprintf('    trSize = %4i, num. classes = %2i, OA = %5.2f %%\n', ...
                    size(trnSet,1), length(classes), res.OA);
    
    % Stop in last iteration (except the special case when only one iteration is requested)
    if length(options.iterVect) ~= 1 && ptsidx == length(options.iterVect)
        break
    end
    
    
    %%%% Here begins the Active Selection algorithm
    
    % 1. Obtain predictions on cndSet for ranking
    
    disp('  Ranking ...')
    
    if strcmpi(options.uncertainty, 'EQB')
        predMatrix = zeros(size(cndSet,1), options.numModels);
        
        % Build predition matrix for EQB running different 'perm' permutations
        fprintf('  EQB model ')
        for i = 1:options.numModels
            fprintf(' %02d', i)
            % New training set
            c = randperm(size(trnSet,1))';
            shuffledTrnSet = trnSet(c(1:ceil(options.pct*length(c))),:);
            
            % SVM training of i-th SVM
            if strcmpi(options.model,'SVM')
                ALtrain(shuffledTrnSet, options.modelParameters, num_of_classes, modelname, rundir);
            end
            
            % Prediction
            predMatrix(:,i) = ...
                ALpredict(options.model, shuffledTrnSet, cndSet, modelname, num_of_classes, rundir);
        end
        fprintf('\n')
            
    elseif ~strcmpi(options.uncertainty, 'random')
        % Predictions on cndSet
        [labels distances] = ...
            ALpredict(options.model, trnSet, cndSet, modelname, num_of_classes, rundir);
    end
    
    
    % 2. Use one the following AL methods to rank the predictions
    
    % Samples to add to training set
    samp2add = diffVect(ptsidx); % options.iterVect(ptsidx+1) - options.iterVect(ptsidx);
    
    switch upper(options.uncertainty)
        
        case 'RANDOM' % Random sampling
            ptsList = randperm(size(cndSet,1))';
            criterion{ptsidx} = 1;
            
        case 'MS' % Margin sampling
            yy = min(abs(distances),[],2);
            [val ptsList] = sortrows(yy);
            criterion{ptsidx} = yy;         
            
        case 'MCLU' % Multiclass Level Uncertanty
            distances = sort(distances,2);
            yy = distances(:,end) - distances(:,end-1);
            [val ptsList] = sortrows(yy);
            criterion{ptsidx} = yy;
            
        case 'EQB'
            %fprintf('  estimating entropies ...\n')
            % Estimate entropy
            CT = hist(predMatrix',classes) ./ options.numModels;
            entropy = log10(CT);
            entropy(isinf(entropy)) = 0; % remote Inf's
            entropy = - sum(CT .* entropy);
            % If options.normEQB ~= 0 compute normalized entropy
            if options.normEQB
                CTcount = log(sum(CT > 0)); % log of non-zero elements
                CTcount(CTcount == 0) = 1; % avoid divisons by zero
                entropy = entropy ./ CTcount;
            end
                        
            %[val ptsList] = sort(-entropy);
            %criterion{ptsidx} = -entropy;
            
            % Sort taking (randomly) into account elements with the same entropy
            c = randperm(length(entropy));
            [val ptsList] = sortrows([-entropy' c'],[1,2]);
            criterion{ptsidx} = -entropy';
            
        % Lexie: MV implementation
        case 'MULTIVIEW'
            svmDir = './MV_SVMs';
            if ~exist(svmDir, 'dir')
                mkdir(svmDir)
            end
            %fprintf('  Current step: %d\n', ptsidx)
            ptsList = amdMV(trnSet, cndSet, num_of_classes, options.viewsVector, ...
                                            samp2add, svmDir, nfolds, sigmas, Cs);
            % amdMV can return less samples than requested, fix it!
            samp2add = length(ptsList);
    end
    
    switch upper(options.diversity)
        % ABD diversity criterion
        case 'ABD'
            ABDcand = ptsList(1:min(samp2add*100,end));
            crit = criterion{ptsidx,1}(ptsList(1:min(samp2add*100,end)));
            if strcmpi(options.model, 'LDA')
                ABDopts.kern = 'lin';
                ABDopts.sigma = 0;
            else
                ABDopts.kern = 'rbf';
                ABDopts.sigma = options.modelParameters.stdzFin;
            end
            yes = ABD_criterion([cndSet(ABDcand,:) ABDcand], crit, samp2add*10 , ABDopts);
            % Re-create ptsList using 'yes'
            ptsList = [yes(:,end) ; setdiff(ptsList,yes(:,end))];
            
        case 'CSV'
            disp('Not implemented yet.')
            return
            
        case 'MANIFOLD'
            disp('Not implemented yet.')
            return
    end
    
    if length(options.iterVect) == 1
        % When called to run only one iteration
        sampList = ptsList;    
    
    % Add selected points from ptsList to trnSet and remove them from cndSet
    else
        % Add selected points to trnSet and remove them from cndSet
        ptsList = ptsList(1:samp2add);
        trnSet = [trnSet ; cndSet(ptsList,:)];
        cndSet(ptsList,:) = [];
        
        % Build sampList vector: The indexes in ptsList refer to cndSet, but cndSet is
        % modified, thus doing this is wrong
        %
        %    sampList(sampSize+1:sampSize+samp2add) = ptsList;
        %
        % The indexes in sampList should refer to the original cndSet. To obtain them we
        % use an auxiliary vector, remPtsList, which has the indexes of the original
        % cndSet.
        
        % 1. Take the indexes referred to the original cndSet saved in remPtsList.
        sampList(sampSize+1:sampSize+samp2add) = remPtsList(ptsList);
        sampSize = sampSize + samp2add;
        % 2. Remove added points from remPtsList
        remPtsList(ptsList) = [];
    end
    
end

% Fix sampList size: For instance, in multiview can be less than expected
sampList = sampList(1:sampSize);

modelParameters = options.modelParameters;
