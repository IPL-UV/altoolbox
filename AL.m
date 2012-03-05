function [tstErr, predictions, modelParams, criterion] = ...
    AL(method, trnSet, trnPool, tstSet, iterVect, num_of_classes, pct, perm, nEQB)

% Active Learning
%
% inputs:   - method: active learning method (see later)
%           - trnSet: starting train set
%           - trnPool: candidates
%           - tstSet: testSet for error computation
%           - iterVect: vector with samples to add at each iteration
%                       ex: 10:20:90, 5 iterations adding 20 samples each
%           - num_of_classes: number of classes
%           - EQB parameters:
%             - pct: percentage of training samples in weak classifiers
%             - perm: number of weak classifiers
%             - nEQB:   - 0: without normalization [Tuia et al., TGRS, 2009]
%                       - 1: with normalization [Copa et al, SPIE, 2010]
%
% outputs:  - tstErr: errors over test set (iter x 1)
%           - predictions: test set predictions
%           - modelParams: SVM hyper-parameters
%           - ptsList: list of samples selected by the AL method
%           - criterion: criterion followed by the AL method to select the samples
%
%           * note: ptsList and criterion are of the last iteration only!
%
% method can be:
%   - 'RS': random sampling
%   - 'MS' / 'MS_ABD': margin sampling (Schohn 2000) / MS with ABD criterion
%   - 'MCLU' / 'MCLU_ABD': multiclass level uncertainty / MCLU + ABD
%   - 'MMD': minimal mean distances between [-1,+1]
%   - 'MCLU_OPC': MCLU selecting exactly one sample per class (using predictions)
%
% by Devis Tuia, JoRdI (2007-12)
%
% See also ALToolbox

% LDA / SVM parameters
rundir = sprintf('./run_%s', method);
if strcmpi(method,'EQB_LDA')
    modelname = '';
    stdzFin = 0;
    costFin = 0;
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
tstErr = zeros(numel(iterVect),2);
predictions = zeros(size(tstSet,1), numel(iterVect));

% -----------------------------------------------------------------------------
% Main loop: iterVect is essentially the number of samples to test
for ptsidx = 1:length(iterVect)
    
    % SVM training
    if strcmpi(method,'EQB_LDA')
        modelname = '';
    else
        % Repeats grid search at iterations 1, 10
        if  ptsidx == 1 || ptsidx == 11
            modelParams = GridSearch_Train_CV(trnSet,num_of_classes,sigmas,Cs,nfolds,rundir);
        end
        % Training
        ALtrain(trnSet, modelParams, num_of_classes, modelname, rundir);
    end
    
    % Predictions on test set
    %disp('  Testing ...')
    
    % Predict in blocks to handle deal with test sets
    predictions(:,ptsidx) = ALpredict(method, trnSet, tstSet, modelname, rundir);
    
    % Classes in current training set
    classes = unique(trnSet(:,end));
    
    % compute test error and Kappa
    res = assessment(tstSet(:,end),predictions(:,ptsidx),'class');
    tstErr(ptsidx,1) = res.OA;
    tstErr(ptsidx,2) = res.Kappa;
    fprintf('    trSize = %4i, num. classes = %2i, OA = %5.2f %%\n', ...
                    size(trnSet,1), length(classes), res.OA);
    
    % Stop in last iteration
    if ptsidx == length(iterVect)
        break
    end
    
    
    %%%% Here begins the Active Selection algorithm
    
    % 1. Obtain predictions on trnPool for ranking
    
    disp('  Ranking ...')
    
    if strcmpi(method, 'EQB_LDA') || strcmpi(method, 'EQB_SVM')
        predMatrix = zeros(size(trnPool,1), perm);
        
        % Build predition matrix for EQB running different 'perm' permutations
        fprintf('  perm ')
        for i = 1:perm
            fprintf(' %02d', i)
            % New training set
            c = randperm(size(trnSet,1))';
            shuffledTrnSet = trnSet(c(1:ceil(pct*length(c))),:);
            
            % SVM training of i-th SVM
            if ~strcmpi(method,'EQB_LDA')
                ALtrain(shuffledTrnSet, modelParams, num_of_classes, modelname, rundir);
            end
            
            % Prediction
            predMatrix(:,i) = ALpredict(method, shuffledTrnSet, trnPool, modelname, rundir);
        end
        fprintf('\n')
            
    elseif ~strcmpi(method, 'RS')
        % Prediction on trnPool
        [labels distances] = ALpredict(method, trnSet, trnPool, modelname, rundir);
    end
    
    
    % 2. Use one the following AL methods to rank the predictions
    
    switch method
        
        case 'RS' % Random sampling
            ptsList = randperm(size(trnPool,1));
            criterion = ones(size(trnPool,1),1);
            
        case {'MS','MS_ABD'} % Margin sampling
            yy = min(abs(distances),[],2);
            [val ptsList] = sortrows(yy);
            criterion = yy;
            
        case {'MCLU','MCLU_ABD'} % Multiclass Level Uncertanty
            distances = sort(distances,2);
            yy = distances(:,end) - distances(:,end-1);
            [val ptsList] = sortrows(yy);
            criterion = yy;
            
        case 'MMD' % Minimal mean distances between [-1,+1]
            newdist = zeros(size(distances,1), 1);
            distances = abs(distances);
            for dd = 1:size(distances,1)
                newdist(dd) = mean(distances( distances(dd,:) < 1 ));
            end
            [val ptsList] = sortrows(newdist);
            criterion = newdist;
            
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
            criterion = distances(:,end) - distances(:,end-1);
            
        case {'EQB_LDA','EQB_SVM'}
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
            criterion = -entropy;
    end
    
    % Samples to add to training set
    samp2add = iterVect(ptsidx+1) - iterVect(ptsidx);
    
    % ABD criterion for MS or MCLU
    if strcmpi(method,'MS_ABD') || strcmpi(method,'MCLU_ABD')
        cand = ptsList(1:min(samp2add*100,end),:);
        
        options.kern = 'rbf';
        options.sigma = modelParams.stdzFin;
        yes = ABD_criterion([trnPool(cand,:) cand], samp2add*10 , options);
        
        % Re-create ptsList using 'yes'
        ptsList = [yes ; setdiff(ptsList,yes)];
    end
    
    % Add selected points from ptsList to trnSet and remove them from trnPool
    ptsNoList = ptsList((samp2add+1):end);
    ptsList   = ptsList(1:samp2add);
    trnSet    = [trnSet ; trnPool(ptsList,:)];
    trnPool   = trnPool(ptsNoList,:);
    
end
