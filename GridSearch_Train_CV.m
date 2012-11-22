function [modelParameters tab] = GridSearch_Train_CV(trainInput, ncl, sigma, cost, N, rundir)

% function [modelParameters tab] = GridSearch_Train_CV(trainInput, ncl, sigma, cost, N, rundir)
%
% Crossvalidation error for SVM parameters selection
% by Michele VOlpi, University of Lausanne, 2010.
%
% See also ALtoolbox

[s1,s2] = size(trainInput);
shake = randperm(s1)';
trainInput = sortrows([trainInput shake], s2+1);
trainInput = trainInput(:,1:s2);
x = trainInput(:,1:s2-1);
y = trainInput(:,s2);

fprintf('  Adjusting SVM parameters ...')

iter = 1;
tab = zeros(length(sigma)*length(cost),3);
cmdval = sprintf('multisvm --val %s/model_GS %s/valSet.txt -dir %s', rundir, rundir, rundir);
if ~ispc, cmdval = ['./' cmdval]; end
for c = cost
    for s = sigma
        tabCV = zeros(N,1);
        cmdtrn = sprintf('multisvm %s/trnSet_init.txt %s/model_GS %i -c %i -std %f -dir %s', ...
                    rundir, rundir, ncl, c, s*sqrt(2), rundir);
        if ~ispc, cmdtrn = ['./' cmdtrn]; end
        for k = 1:N
            [xapp,yapp,xtest,ytest] = n_fold(x,y,N,k);

            trainCV = [xapp yapp];
            valCV = [xtest ytest];

            enteteTrn = [size(trainCV,1) size(trainCV,2)];
            dlmwrite([rundir '/trnSet_init.txt'], enteteTrn, 'delimiter', ' ', 'precision', 8);
            dlmwrite([rundir '/trnSet_init.txt'], trainCV, 'delimiter', ' ', '-append');

            enteteVal = [size(valCV,1) size(valCV,2)];
            dlmwrite([rundir '/valSet.txt'], enteteVal, 'delimiter', ' ', 'precision', 8);
            dlmwrite([rundir '/valSet.txt'], valCV, 'delimiter', ' ', '-append');

            [status,result] = system(cmdtrn);
            if status ~= 0
                disp(result)
                error('ALToolbox:GridSearch_Train_CV:multisvm','Error running multisvm')
            end
            [status,result] = system(cmdval);
            if status ~= 0
                disp(result)
                error('ALToolbox:GridSearch_Train_CV:multisvm','Error running multisvm')
            end

            %err = textread([rundir '/the_class_err.dat']);
            str = regexp(result,'# ([\.\d]+)%','tokens');
            tabCV(k,1) = str2double(str{1});
        end

        tab(iter,:) = [s c mean(tabCV)];
        %fprintf('------------------ %8.2f %8.2f %6.2f \n',tab(iter,:));
        clear tabCV
        iter = iter + 1;
    end
end

[vv ii] = min(tab(:,3));

fprintf('  %5.2f %5.2f %5.2f\n', tab(ii,:))

modelParameters.stdzFin = tab(ii,1);
modelParameters.costFin = tab(ii,2);
