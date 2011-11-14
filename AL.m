%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active Learning methods
%
% inputs:   - method: active learning method (see later)
%           - trnSet: starting train set
%           - trnPool: candidates
%           - tstSet: testSet for error computation
%           - iterVect: vector with iterations (samples to test)
%           - pts2add: number of pts to add at each iteration
%           - num_of_classes: number of classes
%           - EQB parameters:
%             - pct: percentage of subtraining sets
%             - perm: permutations (subsets) to try
%             - nEQB: 0: ---- / 1: ----
%
% outputs:  - tstErr: errors over test set (iter x 1)
%           - predictions: test set predictions
%           - stdzFin costFin : kernel parameters
%
% method can be:
%   - 'RS': random sampling
%   - 'MS' / 'MS_ABD': margin sampling (Schohn 2000) / MS with ABD criterion
%   - 'MCLU' / 'MCLU_ABD': multiclass level uncertainty / MCLU + ABD
%   - 'MMD': minimal mean distances between [-1,+1]
%   - 'MCLU_OPC': MCLU selecting exactly one sample per class (using predictions)
%
% by Devis Tuia (2007), JoRdI (2011)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is what original EQB_ function returned
%function [tstErr, predictions, stdzFin, costFin, trnSet, trnPool, nSV] = ...

function [tstErr, predictions, stdzFin, costFin] = ...
    AL(method, trnSet, trnPool, tstSet, iterVect, pts2add, num_of_classes, pct, perm, nEQB)

% Test set and training parameters for SVM
if ~strcmpi(method,'EQB_LDA')
    % Running directory for multisvm
    rundir = sprintf('./run_%s', method);
    if ~exist(rundir,'dir')
        mkdir(rundir)
    end
    % Test set
    enteteTstSet = [size(tstSet,1) size(tstSet,2)];
    dlmwrite([rundir '/tst.txt'], enteteTstSet, ' ');
    dlmwrite([rundir '/tst.txt'], tstSet, 'delimiter', ' ', '-append');
    % SVM training parameters
    nfolds = 3;
    sigmas = logspace(-1,2,7);
    Cs = logspace(0,3,7);
end

% Results
tstErr = zeros(numel(iterVect),2);
predictions = zeros(size(tstSet,1), numel(iterVect));

% -----------------------------------------------------------------------------
% Main loop: iterVect is essentially the number of samples to test
ptsidx = 1;
for iter = iterVect
    
    if strcmpi(method,'EQB_LDA')
        labels = classify(tstSet(:,1:end-1),trnSet(:,1:end-1),trnSet(:,end));
    else
        % refait le grid search toutes les 10 iterations
        if iter == 10 || iter == 100 % || iter == 1000        
            [stdzFin, costFin] = GridSearch_Train_CV(trnSet,num_of_classes,sigmas,Cs,nfolds,rundir);
        end

        % Training set
        enteteSTrnSet = [size(trnSet,1) size(trnSet,2)];
        dlmwrite([rundir '/TrnSetShuffled.txt'], enteteSTrnSet, ' ');
        dlmwrite([rundir '/TrnSetShuffled.txt'], trnSet, 'delimiter', ' ', '-append');

        % entranement + test du i-me svm
        modelname = sprintf('%s/modelTestBoot_Schohn', rundir);

        cmdtrn = sprintf('./multisvm %s/TrnSetShuffled.txt %s %d -c %f -std %f -dir %s', ...
                        rundir, modelname, num_of_classes, costFin, stdzFin * sqrt(2), rundir);

        cmdval = sprintf('./multisvm --val %s %s/tst.txt -dir %s', modelname, rundir, rundir);

        [status,result] = system(cmdtrn);
        if status ~= 0
            disp(result)
            error('ALToolbox:AL:multisvm','Error running multisvm')
        end
        [status,result] = system(cmdval);
        if status ~= 0
            disp(result)
            error('ALToolbox:AL:multisvm','Error running multisvm')
        end

        labels = textread([rundir '/predictions.dat']);    
    
    end
    
    % compute test error and Kappa
    res = assessment(tstSet(:,end),labels(:,1),'class');
    tstErr(ptsidx,1) = res.OA;
    tstErr(ptsidx,2) = res.Kappa;
    fprintf('iteration %3i, trSize = %4i, Ncl = %2i OA = %2.2f\n', ...
                iter, size(trnSet,1), size(unique(trnSet(:,end)),1), res.OA);
    
    %if ptsidx == 1 || mod(ptsidx,10) == 0
        predictions(:,ptsidx) = labels(:,1);
    %end
    
    % Stop when we are in last iteration
    if iter == iterVect(end)
        break
    end
    
    
    %%%% Here begins the Active Selection algorithm
    
    % Current classes
    classes = unique(trnSet(:,end));
    
    % Prepare SVM prediction on pool samples
    if ~strcmpi(method,'EQB_LDA')
        enteteSTrnSet = [size(trnPool,1) size(trnPool,2)];
        dlmwrite([rundir '/trnPool.txt'], enteteSTrnSet, ' ');
        dlmwrite([rundir '/trnPool.txt'], trnPool, 'delimiter', ' ', '-append');
        cmdval = sprintf('./multisvm --val %s %s/trnPool.txt -dir %s', modelname, rundir, rundir);
    end
    
    if strcmpi(method,'EQB_LDA') || strcmpi(method,'EQB_SVM')
        % Build predition matrix for EQB running different 'perm' permutations
        predMatrix = zeros(size(trnPool,1),perm);
        fprintf('  perm ')
        for i = 1:perm
            fprintf(' %02d', i)
            % New training set
            c = randperm(size(trnSet,1))';
            shuffledTrnSet(:,:) = trnSet(c(1:ceil(pct*size(c,1)),1),:);
            clear c
            
            if strcmpi(method,'EQB_LDA')
                pred = classify(trnPool(:,1:end-1),shuffledTrnSet(:,1:end-1),shuffledTrnSet(:,end));
                predMatrix(:,i) = pred;
            else
                % Training set
                enteteSTrnSet = [size(shuffledTrnSet,1) size(shuffledTrnSet,2)];
                dlmwrite([rundir '/TrnSetShuffled.txt'], enteteSTrnSet, ' ');
                dlmwrite([rundir '/TrnSetShuffled.txt'], shuffledTrnSet, 'delimiter', ' ', '-append');
                % Training + prediction of i-th SVM
                [status,result] = system(cmdtrn);
                if status ~= 0
                    disp(result)
                    error('ALToolbox:AL:multisvm','Error running multisvm')
                end
                [status,result] = system(cmdval);
                if status ~= 0
                    disp(result)
                    error('ALToolbox:AL:multisvm','Error running multisvm')
                end
                pred = textread([rundir '/predictions.dat']);
                predMatrix(:,i) = pred(:,1);
            end
            clear pred errTst shuffledTrnSet
        end
        fprintf('\n')
    else
        % SVM prediction on pool set
        [status,result] = system(cmdval);
        if status ~= 0
            disp(result)
            error('ALToolbox:AL:multisvm','Error running multisvm')
        end
        labels = textread([rundir '/predictions.dat']);
        distances = labels(:,2:end);
        labels = labels(:,1);
    end
    
    
    switch method
        
        case 'RS' % Random sampling
            ptsList = randperm(size(trnPool,1));
            
        case {'MS','MS_ABD'} % Margin sampling
            yy = min(abs(distances),[],2);
            [val ptsList] = sortrows(yy);
            
        case {'MCLU','MCLU_ABD'} % Multiclass Level Uncertanty
            distances = sort(distances,2);
            yy = distances(:,end) - distances(:,end-1);
            [val ptsList] = sortrows(yy);
            
        case 'MMD' % Minimal mean distances between [-1,+1]
            newdist = zeros(size(distances,1), 1);
            distances = abs(distances);
            for dd = 1:size(distances,1)
                newdist(dd) = mean(distances( distances(dd,:) < 1 ));
            end
            [val ptsList] = sortrows(newdist);
            
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
            
    end
    
    % ABD criterion for MS or MCLU
    if strcmpi(method,'MS_ABD') || strcmpi(method,'MCLU_ABD')
        cand = ptsList(1:10*pts2add(ptsidx),:);

        options.kern = 'rbf';
        options.sigma = stdzFin;
        yes = ABD_criterion([trnPool(cand,:) cand], pts2add(ptsidx) , options);

        % Re-create ptsList using 'yes'
        ptsList = [yes ; setdiff(ptsList,yes)];        
    end
    
    % Add selected points from ptsList to trnSet and remove them from trnPool
    ptsNoList = ptsList((pts2add(ptsidx)+1):end);
    ptsList   = ptsList(1:pts2add(ptsidx));
    trnSet    = [trnSet ; trnPool(ptsList,:)];
    trnPool   = trnPool(ptsNoList,:);        
    
    ptsidx = ptsidx + 1;
    
    clear distances
    
end
