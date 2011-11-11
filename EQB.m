% Active Learning for training sets of remote sensing images
%
% [tstErr, predictions, trnSet, trnNo, nSV, stdzFin, costFin] = ...
%     EQB(trnSet, trnNo, tstSet, maxIter, pts2add, num_of_classes, pct, perm, nEQB,model)
%
% inputs:     - trnSet: initial training set, with nxm variables and nx1 labels stacked
%             - trnNo : candidates to be added in the trnSet, with nxm variables and nx1 labels stacked
%             - valSet: validation set for grid search of parameters nxm variables, nx1 labels
%             - tstSet: testSet to compute errors, with nxm data and nx1 labels stacked
%             - maxIter = number of iterations
%             - perm: number of SVMs for the bagging
%             - pts2add: number of points to add at each iteration
%             - pct: percent of data to select during the bagging
%             - nEQB: normalization (nEQB, Copa et al. 2010). Default '0'.
%                     Avoids oversampling in areas of uncertainty among many classes
%             - model: model to use ('SVM','LDA')
%
% outputs:    - tstErr = OA and Kappa curves
%             - pred = predictions at each 20 iterations
%             - trnSet = final trn Set
%             - trnNo = final candidates
%             - nSV: number of SV per class and iteration (SVM only)
%			  - costFin stdzFin = kernel parameters (SVM only)
%
% by Devis Tuia (2010), JoRdI (2011)

function [tstErr, predictions, trnSet, trnNo,nSV, stdzFin, costFin] = ...
    EQB(trnSet, trnNo, tstSet, iterVect, pts2add, num_of_classes, pct, perm, nEQB, model, rundir)

if ~exist('rundir','var')
    rundir = './run_eqb';
end
if ~strcmpi(model,'LDA') && ~exist(rundir,'dir')
    mkdir(rundir)
end
if ~exist('nEQB','var')
    nEQB = 0;
end
if ~exist('model','var')
    model = 'SVM';
end

switch model
    case 'SVM'
        enteteTstSet = [size(tstSet,1) size(tstSet,2)];
        dlmwrite([rundir '/tst.txt'], enteteTstSet, ' ');
        dlmwrite([rundir '/tst.txt'], tstSet, 'delimiter', ' ', '-append');
    case 'LDA'
        nSV = 0;
end

tstErr = zeros(numel(iterVect),2); %zeros(maxIter,2);
predictions = zeros(size(tstSet,1), numel(iterVect)); %[];
nSV = zeros(length(iterVect),num_of_classes);

nfolds = 3;
sigmas = logspace(-1,2,7);
Cs = logspace(0,3,7);

ptsidx = 1;

for iter = iterVect
    
    switch model        
        case 'SVM'            
            % refait le grid search toutes les 10 iterations
            if iter == 10 || iter == 100 % || iter == 1000
                [stdzFin, costFin] = GridSearch_Train_CV(trnSet,num_of_classes,sigmas,Cs,nfolds,rundir);
            end
            
            enteteSTrnSet = [size(trnSet,1) size(trnSet,2)];
            dlmwrite([rundir '/TrnSetShuffled_m_02.txt'], enteteSTrnSet, ' ');
            dlmwrite([rundir '/TrnSetShuffled_m_02.txt'], trnSet, 'delimiter', ' ', '-append');
            
            % entra nement + test du i- me svm
            modelname = sprintf('%s/modelEntropy%d_p%d', rundir, iter, perm);
            
            cmdtrn = sprintf('./multisvm %s/TrnSetShuffled_m_02.txt %s %d -c %f -std %f -dir %s', ...
                            rundir, modelname, num_of_classes, costFin, stdzFin * sqrt(2), rundir);
            
            cmdval = sprintf('./multisvm --val %s %s/tst.txt -dir %s', modelname, rundir, rundir);

            % train model and use it on test
            [status,result] = system(cmdtrn);
            if status ~= 0
                disp(result)
                error('ALToolbox:EQB:multisvm','Error running multisvm')
            end
            [status,result] = system(cmdval);
            if status ~= 0
                disp(result)
                error('ALToolbox:EQB:multisvm','Error running multisvm')
            end
            
            labels = textread([rundir '/predictions.dat']);
            
        case 'LDA'
            labels = classify(tstSet(:,1:end-1),trnSet(:,1:end-1),trnSet(:,end));
            
    end
    
    % compute test error and Kappa
    res = assessment(tstSet(:,end),labels(:,1),'class');
    tstErr(ptsidx,1) = res.OA;
    tstErr(ptsidx,2) = res.Kappa;
    fprintf('iteration %3i, trSize = %4i, Ncl = %2i OA = %2.2f\n',...
                iter, size(trnSet,1), size(unique(trnSet(:,end)),1), res.OA);
    
    %if iter == 1 || mod(iter,20) == 0
        predictions(:,ptsidx) = labels(:,1);
    %end
    
    % Stop when we are in last iteration
    if iter == iterVect(end)
        break
    end
    
    
    %%%% Here is where Active Selection begins
    
    % Prepare trnNo set for EQB
    if strcmpi(model,'SVM')
        enteteTrnNo = [size(trnNo,1) size(trnNo,2)];
        dlmwrite([rundir '/trnNoBoot_m_02.txt'], enteteTrnNo, ' ');
        dlmwrite([rundir '/trnNoBoot_m_02.txt'], trnNo, 'delimiter', ' ', '-append');
    end
    
    predMatrix = zeros(size(trnNo,1),perm);
    % prise de 75% des donn es d'entrainement "perm" fois
    fprintf('  perm ')
    for i = 1:perm
        
        fprintf(' %02d', i)
        
        c = randperm(size(trnSet,1))';
        shuffledTrnSet(:,:) = trnSet(c(1:ceil(pct*size(c,1)),1),:);
        clear c
        
        %modelname = sprintf('modelEntropy%d_p%d', iter, perm);
        
        switch model
            
            case 'SVM'
                enteteSTrnSet = [size(shuffledTrnSet,1) size(shuffledTrnSet,2)];
                dlmwrite([rundir '/TrnSetShuffled_m_02.txt'], enteteSTrnSet, ' ');
                dlmwrite([rundir '/TrnSetShuffled_m_02.txt'], shuffledTrnSet, 'delimiter', ' ', '-append');                
                                
                % version avec le calcul de l'entropie sur les predictions
                cmdval = sprintf('./multisvm --val %s %s/trnNoBoot_m_02.txt -dir %s', modelname, rundir, rundir);
                
                % entra nement + test du i- me svm
                [status,result] = system(cmdtrn);
                if status ~= 0
                    disp(result)
                    error('ALToolbox:EQB:multisvm','Error running multisvm')
                end
                [status,result] = system(cmdval);
                if status ~= 0
                    disp(result)
                    error('ALToolbox:EQB:multisvm','Error running multisvm')
                end
                
                pred = textread([rundir '/predictions.dat']);
                predMatrix(:,i) = pred(:,1);
                
            case 'LDA'
                pred = classify(trnNo(:,1:end-1),shuffledTrnSet(:,1:end-1),shuffledTrnSet(:,end));
                predMatrix(:,i) = pred;
        end
        
        clear pred errTst shuffledTrnSet
    end
    fprintf('\n')
    
    if strcmpi(model,'SVM')
        for cl = 1:num_of_classes
            a = sprintf([rundir '/SV%u.dat'],cl-1);
            SV = textread(a);
            nSV(iter,cl) = size(SV,1);
        end
    end
    
    entropy = zeros(size(predMatrix,1),1);
    % Entropy computation
    for k = 1:size(predMatrix,1)
        CT = (crosstab(predMatrix(k,:))./perm)';
        if nEQB
            entropy(k,1) = max(-1*sum(CT.*log10(CT))/log(size(CT,2)),0);
        else
            entropy(k,1) = -1 * sum(CT.*log10(CT));
        end
    end
    
    % plots of entropy
    %     if iter == 1
    %
    %         figure(1)
    %         scatter(trnNo(:,1),trnNo(:,2),18,entropy(:,1),'f')
    %         title('entropy')
    %         grid off
    %         figure(2)
    %         contourf(reshape(entropy,110,110))
    %         keyboard
    %     end
    
    % ajoute l'ID des pts apres les predictions
    predMatrix(:,size(predMatrix,2)+1) = (1:size(predMatrix,1))';
    
    % selection des points
    predMatrix  = [entropy predMatrix];
    sPredMatrix = -1 * sortrows(-1*predMatrix,1);
    %ptsList = sPredMatrix(1:pts2add,size(sPredMatrix,2));
    ptsList = sPredMatrix(1:pts2add(ptsidx),size(sPredMatrix,2));
    
    % ajout des points
    trnSet = [trnSet ; trnNo(ptsList,:)];
    
    % retire les pts ajout_s du dataset "No"
    %ptsNoList = sPredMatrix(pts2add+1:size(sPredMatrix,1),size(sPredMatrix,2));
    ptsNoList = sPredMatrix((pts2add(ptsidx)+1):size(sPredMatrix,1),size(sPredMatrix,2));
    trnNo = trnNo(ptsNoList,:);
    
    ptsidx = ptsidx + 1;
    
    clear errTst predMatrix sPredMatrix entropy CT ptsList
    
end
