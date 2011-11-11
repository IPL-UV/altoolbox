%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active learning with choice of the best points by
% - distance to the margin (Schohn 2000)
%
% inputs:   - trnSet: starting train set
%           - trnNo: candidates
%           - tstSet: testSet for error computation
%           - maxIter : number of iterations
%           - num_of_classes: number of classes
%           - pts2add: number of pts to add at every iteration
%           - mode: 0 = margin sampling, 1 = MCLU
%
% outputs:  - tstErr: erreurs sur le testSet (iter x 1)
%           - stdzFin costFin : kernel parameters
%
% by Devis Tuia (2007), JoRdI (2011)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [tstErr, predictions, stdzFin, costFin] = ...
    MS(trnSet, trnNo, tstSet, iterVect, pts2add, num_of_classes, mode, rundir)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('rundir','var')
    rundir = './run_ms';
end
if ~exist(rundir,'dir')
    mkdir(rundir)
end

% Test set
enteteTstSet = [size(tstSet,1) size(tstSet,2)];
dlmwrite([rundir '/tst.txt'], enteteTstSet, ' ');
dlmwrite([rundir '/tst.txt'], tstSet, 'delimiter', ' ', '-append');

tstErr = zeros(numel(iterVect),2);
predictions = zeros(size(tstSet,1), numel(iterVect));

nfolds = 3;
sigmas = logspace(-1,2,7);
Cs = logspace(0,3,7);

if mode == 3
    % Extract classes
    classes = unique(trnNo(:,end));
end

ptsidx = 1;

for iter = iterVect
    % refait le grid search toutes les 10 iterations
    if iter == 10 || iter == 100 % || iter == 1000        
        [stdzFin, costFin] = GridSearch_Train_CV(trnSet,num_of_classes,sigmas,Cs,nfolds,rundir);
        %stdzFin = 44.7
        %costFin = 1e3
    end
    
    % Training set
    enteteSTrnSet = [size(trnSet,1) size(trnSet,2)];
    dlmwrite([rundir '/TrnSetShuffled_m_02.txt'], enteteSTrnSet, ' ');
    dlmwrite([rundir '/TrnSetShuffled_m_02.txt'], trnSet, 'delimiter', ' ', '-append');
    
    % entranement + test du i-me svm
    
    modelname = sprintf('%s/modelTestBoot_Schohn%d', rundir, iter);
    
    cmdtrn = sprintf('./multisvm %s/TrnSetShuffled_m_02.txt %s %d -c %f -std %f -dir %s', ...
                    rundir, modelname, num_of_classes, costFin, stdzFin * sqrt(2), rundir);
    
    cmdval = sprintf('./multisvm --val %s %s/tst.txt -dir %s', modelname, rundir, rundir);
    
    [status,result] = system(cmdtrn);
    if status ~= 0
        disp(result)
        error('ALToolbox:MS:multisvm','Error running multisvm')
    end
    [status,result] = system(cmdval);
    if status ~= 0
        disp(result)
        error('ALToolbox:MS:multisvm','Error running multisvm')
    end
    
    labels = textread([rundir '/predictions.dat']);    
    
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
    
    % Predict on all remaining unlabeled samples
    enteteSTrnSet = [size(trnNo,1) size(trnNo,2)];
    dlmwrite([rundir '/trnNo.txt'], enteteSTrnSet, ' ');
    dlmwrite([rundir '/trnNo.txt'], trnNo, 'delimiter', ' ', '-append');
    
    cmdval = sprintf('./multisvm --val %s %s/trnNo.txt -dir %s', modelname, rundir, rundir);
    [status,result] = system(cmdval);
    if status ~= 0
        disp(result)
        error('ALToolbox:MS:multisvm','Error running multisvm')
    end
    
    labels = textread([rundir '/predictions.dat']);
    distances = labels(:,2:end);
    labels = labels(:,1);
    
    % RS / MS [+ ABD] / MCLU [+ ABD]
    switch mode
        
        case 10 % Random sampling
            ptsList = randperm(size(trnNo,1));
            
        case {0,4} % Traditional MS [+ ABD]
            yy = min(abs(distances),[],2);
            [val ptsList] = sortrows(yy);
            
        case {1,5} % MCLU (Multiclass Level Uncertanty) [+ ABD]
            distances = sort(distances,2);
            yy = distances(:,end) - distances(:,end-1);
            [val ptsList] = sortrows(yy);
            
        case 2 % Minimal mean distances between [-1,+1]
            newdist = zeros(size(distances,1), 1);
            distances = abs(distances);
            for dd = 1:size(distances,1)
                newdist(dd) = mean(distances( distances(dd,:) < 1 ));
            end
            [val ptsList] = sortrows(newdist);
            
        case 3 % MCLU selecting exactly one from each class
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
            
    end
    
    % ABD criterion for MS or MCLU
    if mode == 4 || mode == 5
        cand = ptsList(1:10*pts2add(ptsidx),:);

        options.kern = 'rbf';
        options.sigma = stdzFin;
        yes = ABD_criterion([trnNo(cand,:) cand], pts2add(ptsidx) , options);

        % Re-create ptsList using 'yes'
        ptsList = [yes ; setdiff(ptsList,yes)];        
    end
    
    ptsNoList = ptsList((pts2add(ptsidx)+1):end);
    ptsList   = ptsList(1:pts2add(ptsidx));
    trnSet    = [trnSet ; trnNo(ptsList,:)];
    trnNo     = trnNo(ptsNoList,:);        
    
    ptsidx = ptsidx + 1;
    
    clear distances xx s_pts
    
end
