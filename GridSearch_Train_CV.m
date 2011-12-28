function [stdzFin, costFin, tab] = GridSearch_Train_CV(trainInput,ncl,sigma,cost,N,rundir)

% SVM automatic trainer
%
% function [stdzFin, costFin, tab] = GridSearch_Train_CV(trainInput,ncl,sigma,cost,N,rundir)

[s1,s2] = size(trainInput);
shake = randperm(s1)';
trainInput = sortrows([trainInput shake], s2+1);
trainInput = trainInput(:,1:s2);
x = trainInput(:,1:s2-1);
y = trainInput(:,s2);

iter = 1;
% tab = zeros(size(unique(sigma),1)*size(unique(cost),1),N);
tab = zeros(1,N);
cmdval = sprintf('./multisvm --val %s/model_GS %s/valSet.txt -dir %s', rundir, rundir, rundir);
for c = cost
    for s = sigma
        tabCV = zeros(N,1);
        cmdtrn = sprintf('./multisvm %s/trnSet_init.txt %s/model_GS %i -c %i -std %f -dir %s', ...
                    rundir, rundir, ncl, c, s*sqrt(2), rundir);
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
        fprintf('------------------ %8.2f %8.2f %6.2f \n',tab(iter,:));
        clear tabCV
        iter = iter + 1;
    end
end

clear iter
a = find(tab(:,3) == min(tab(:,3)));
bst = tab(a(size(a,1)),:);

% disp(['Proposed:'])
% disp(['s1 C error'])
disp(bst)

stdzFin = tab(a(size(a,1)),1);
costFin = tab(a(size(a,1)),2);
