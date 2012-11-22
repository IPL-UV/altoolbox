% This code is to demo two AL algorithms, MS and MV AMD, using KSC 1 data.
% We also have a base line Random Sampling for comparisons. 
clear; close all;
% KSC1 data set has 176 bands for 3784 samples.
load testKSC1

% We take 50% of data minus 30 samples, which are for initial training 
% (see the below script), as our candidate set.
cndSet = testKSC1(1:1862,:);
testKSC1(1:1862,:) = [];
% test set: tsSet
% we take another disjoint 50% of data as our testing set
tsSet = testKSC1(1:1892,:);
testKSC1(1:1892,:) = [];
% initial traing set : trSet
% 30 samples for the initial training set
trSet = testKSC1;

% 5 samples for each learning step. 120 learning steps here. 
iterVect = 5:5:100;
num_of_classes = length(unique(trSet(:,end)));

% Random sampling
disp('SVM with random sampling');
options.model = 'SVM';
options.uncertainty = 'Random';
options.diversity = 'None';
options.iterVect = iterVect;
options.paramSearchIters = [1 2];

name = sprintf('%s_%s_%s', options.model, options.uncertainty, options.diversity);
[accCurve.(name) predictions.(name), criterion.(name), sampList.(name), modelParameters.(name)] = ...
             AL(trSet, cndSet, tsSet, num_of_classes, options);

% Margin sampling (MS)
disp('SVM with margin sampling active learning');
options.model = 'SVM';
options.uncertainty = 'MS';
options.diversity = 'None';
options.iterVect = iterVect;
options.paramSearchIters = [1 2];

name = sprintf('%s_%s_%s', options.model, options.uncertainty, options.diversity);
[accCurve.(name) predictions.(name), criterion.(name), sampList.(name), modelParameters.(name)] = ...
             AL(trSet, cndSet, tsSet, num_of_classes, options);

% MultiView AMD (MV AMD)
% Multi View settings. Correlation criterion is used in this example to
% subset views (bands).
disp('SVM with MV AMD active Learning ');
viewsVector(1:12)    = 1;
viewsVector(13:32)   = 2;
viewsVector(33:97)   = 3;
viewsVector(98:131)  = 4;
viewsVector(132:176) = 5;
options.model = 'SVM';
options.uncertainty = 'MultiView';
options.diversity = 'None';
options.iterVect = iterVect;
options.viewsVector = viewsVector;
options.paramSearchIters = [1 2];

name = sprintf('%s_%s_%s', options.model, options.uncertainty, options.diversity);
[accCurve.(name) predictions.(name), criterion.(name), sampList.(name), modelParameters.(name)] = ...
             AL(trSet, cndSet, tsSet, num_of_classes, options);

figure
plot(size(trSet,1)+iterVect,accCurve.SVM_Random_None(:,1),'r-');
hold on
plot(size(trSet,1)+iterVect,accCurve.SVM_MS_None(:,1),'b-');
plot(size(trSet,1)+iterVect,accCurve.SVM_MultiView_None(:,1),'k-');
grid on
legend('RS','MS','MV-AMD')
xlabel('Samples in training set')
ylabel('Accuracy [pct]')
