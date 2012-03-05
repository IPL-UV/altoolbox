% AL toolbox demo

% Indian Pines image, 13 out of the 16 classes
% IM are the spectra of the labeled pixels
% CL are the classes
load testAVIRIS.mat

num_of_classes = size(unique(CL),1);
CL = CL-1; % classes must start at 0 for SVMtorch

s = rand('twister');
rand('twister',0);
c = randperm(length(CL))';
rand('twister',s);

tr = [IM(c(1:400),:) CL(c(1:400),:)];
cand = [IM(c(401:8000),:) CL(c(401:8000),:)];
ts = [IM(c(8001:end),:) CL(c(8001:end),:)];

iterVect = 10:10:200;

% RS

% disp('SVM with random sampling');
% [tstErrRS, predictionsRS, modelParamsRS] = ...
%     AL('RS', tr, cand, ts, iterVect, num_of_classes);


options.model = 'SVM';
options.uncertainty = 'random';
options.iterVect = iterVect;

[accCurve, predictions, criterion, sampList, modelParameters] = ...
             ALcore(tr, cand, ts, num_of_classes, options);

return

% AL
disp('SVM with margin sampling active learning');
[tstErrAL, predictionsAL, modelParamsAL] = ...
    AL('MS', tr, cand, ts, iterVect, num_of_classes);

figure
plot(length(tr)+iterVect,tstErrRS(:,1),'r-');
hold on
plot(length(tr)+iterVect,tstErrAL(:,1),'b-');
grid on
legend('RS','MS')
xlabel('Samples in training set')
ylabel('Accuracy [pct]')
