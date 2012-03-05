function [accCurve, predictions, criterion, sampList, modelParameters] = ...
             ALcore(trnSet, cndSet, tstSet, num_of_classes, options)

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
%  	         .uncertainty: 'Random', 'EQB', 'MS', 'MCLU', 'Multiview'
%            .diversity:   'None', 'ABD', 'Manifold'
%            .iterVector:  vector containing number of samples to add at each iteration
%            .model:       base classifier, 'LDA', 'SVM'
%            .pct:         for EQB, percentage of trainig pixels to make sub-trainig sets
%            .numModels:   for EQB, number of models to train
%            .viewsVector: for Multiview, ex: [1 1 1 2 2 3] = 3 views with corresponding
%                          variables
%   - Notes:
%     - MS and MCLU work only if 'SVM' chosen as classifier. If chosing 'LDA', we can use
%       posterior probabilities instead of decision functions to 'simulate' the criteria,
%       but it is a good idea to mix everything
%     - Default options are: {'MS', 'None', 10, 'SVM', 0.6, 4, [1 1 1 2 2 3]}
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
% by Devis Tuia, JoRdI (2007-12)
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
if ~isfield(options, 'viewsVector')
    options.viewsVector = [1 1 1 2 2 3];
end

if strcmpi(options.uncertainty, 'MS') || strcmpi(options.uncertainty, 'MCLU')
    if strcmpi(options.model, 'LDA')
        error([options.uncertainty ' uncertainty does not work with ' options.model])
    end
end

switch options.model
    case {'LDA', 'SVM'}
    otherwise
        error(['Unknown or unimplemented base model: ' options.model])
end

% LDA / SVM parameters
rundir = sprintf('./run_%s_%s', options.model, options.uncertainty);
if strcmpi(options.model,'LDA')
    modelname = '';
    modelParameters.stdzFin = 0;
    modelParameters.costFin = 0;
else
    if ~exist(rundir,'dir')
        mkdir(rundir)
    end
    modelname = sprintf('%s/modelTestBoot_Schohn', rundir);    
    % SVM training parameters
    nfolds = 3;
    sigmas = logspace(-2,1,5);
    Cs = logspace(0,2,5);
end

% Results
accCurve = zeros(numel(options.iterVect),2);
predictions = zeros(size(tstSet,1), numel(options.iterVect));
criterion = cell(numel(options.iterVect),1);

% Reserve memory
diffVect = diff(options.iterVect);
sampList = zeros(1,sum(diffVect));

% Vector with the indexes to fill sampList (this allows to fill it without using
% sampList = [sampList ; <whatever>] 
% idxVect  = [1 cumsumdiffVect];

% Another way is to mantain an index of the vector current size
sampSize = 0;

% -----------------------------------------------------------------------------
% Main loop: iterVect contains the number of samples to add at each iteration
for ptsidx = 1:length(options.iterVect)
    
    % SVM training
    if strcmpi(options.model,'LDA')
        modelname = '';
    else
        % Repeats grid search at iterations 1, 10
        if  ptsidx == 1 || ptsidx == 11
           modelParameters = GridSearch_Train_CV(trnSet,num_of_classes,sigmas,Cs,nfolds,rundir);
        end
        % Training
        ALtrain(trnSet, modelParameters, num_of_classes, modelname, rundir);
    end
    
    % Predictions on test set
    %disp('  Testing ...')
    
    % Predict in blocks to handle deal with test sets
    predictions(:,ptsidx) = ALpredict(options.model, trnSet, tstSet, modelname, rundir);
    
    % Classes in current training set
    classes = unique(trnSet(:,end));
    
    % compute test error and Kappa
    res = assessment(tstSet(:,end),predictions(:,ptsidx),'class');
    accCurve(ptsidx,1) = res.OA;
    accCurve(ptsidx,2) = res.Kappa;
    fprintf('    trSize = %4i, num. classes = %2i, OA = %5.2f %%\n', ...
                    size(trnSet,1), length(classes), res.OA);
    
    % Stop in last iteration
    if ptsidx == length(options.iterVect)
        break
    end
    
    
    %%%% Here begins the Active Selection algorithm
    
    % 1. Obtain predictions on cndSet for ranking
    
    disp('  Ranking ...')
    
    if strcmpi(options.uncertainty, 'EQB')
        predMatrix = zeros(size(cndSet,1), perm);
        
        % Build predition matrix for EQB running different 'perm' permutations
        fprintf('  perm ')
        for i = 1:perm
            fprintf(' %02d', i)
            % New training set
            c = randperm(size(trnSet,1))';
            shuffledTrnSet = trnSet(c(1:ceil(pct*length(c))),:);
            
            % SVM training of i-th SVM
            if strcmpi(options.model,'SVM')
                ALtrain(shuffledTrnSet, modelParameters, num_of_classes, modelname, rundir);
            end
            
            % Prediction
            predMatrix(:,i) = ALpredict(options.model, shuffledTrnSet, cndSet, modelname, rundir);
        end
        fprintf('\n')
            
    elseif ~strcmpi(options.uncertainty, 'random')
        % Predictions on cndSet
        [labels distances] = ALpredict(options.model, trnSet, cndSet, modelname, rundir);
    end
    
    
    % 2. Use one the following AL methods to rank the predictions
    
    switch upper(options.uncertainty)
        
        case 'RANDOM' % Random sampling
            ptsList = randperm(size(cndSet,1));
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
            
        case 'MMD' % Minimal mean distances between [-1,+1]
            yy = zeros(size(distances,1), 1);
            distances = abs(distances);
            for dd = 1:size(distances,1)
                yy(dd) = mean(distances( distances(dd,:) < 1 ));
            end
            [val ptsList] = sortrows(yy);
            criterion{ptsidx} = yy;
            
        case 'MCLU_OPC' % MCLU selecting exactly one from each class (OPC)
            distances = sort(distances,2);
            % Ordered list of differences between maximum distances with predicted label
            xx = [labels sortrows(distances(:,end) - distances(:,end-1))];
            % Select one for each class
            ptsList = 1:size(xx,1);
            for i = 1:numel(classes)
                % Find first element of i-th class
                ptsList(i) = find(xx(:,1) == classes(i), 1);
            end
            % Fill the rest of ptsList with the numbers not already in ptsList
            ptsList((numel(classes)+1):end) = setdiff(1:length(ptsList), ptsList(1:numel(classes)));
            criterion{ptsidx} = distances(:,end) - distances(:,end-1);
            
        case 'EQB'
            %fprintf('  estimating entropies ...\n')
            % Estimate entropy
            CT = hist(predMatrix',classes) ./ perm;
            entropy = log10(CT);
            entropy(isinf(entropy)) = 0; % remote Inf's
            entropy = - sum(CT .* entropy);
            % If nEQB ~= 0 compute normalized entropy
            if nEQB
                CTcount = log(sum(CT > 0)); % log of non-zero elements
                CTcount(CTcount == 0) = 1; % avoid divisons by zero
                entropy = entropy ./ CTcount;
            end
            
            [val ptsList] = sort(-entropy);
            criterion{ptsidx} = -entropy;
    end
    
    % Samples to add to training set
    samp2add = diffVect(ptsidx); % options.iterVect(ptsidx+1) - options.iterVect(ptsidx);
    
    % ABD diversity criterion
    if strcmpi(options.diversity,'ABD')
        cand = ptsList(1:min(samp2add*100,end),:);
        
        options.kern = 'rbf';
        options.sigma = modelParameters.stdzFin;
        yes = ABD_criterion([cndSet(cand,:) cand], samp2add*10 , options);
        
        % Re-create ptsList using 'yes'
        ptsList = [yes ; setdiff(ptsList,yes)];
    end
    
    % Add selected points from ptsList to trnSet and remove them from cndSet
    ptsNoList = ptsList((samp2add+1):end);
    ptsList   = ptsList(1:samp2add);
    trnSet    = [trnSet ; cndSet(ptsList,:)];
    cndSet    = cndSet(ptsNoList,:);
    
    %sampList = [sampList ; ptsList];
    %sampList(idxVect(ptsidx)+1:idxVect(ptsidx+1)) = ptsList;
    sampList(sampSize+1:sampSize+length(ptsList)) = ptsList;
    sampSize = sampSize + length(ptsList);
    
end
