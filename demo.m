%function demo

load testAVIRIS.mat
% indian pines image, 13 out of the 16 classes
%IM are the spectra of the labeled pixels
%CL are the classes

num_of_classes = size(unique(CL),1);
CL = CL-1; %classes must start at 0 for SVMtorch

c = randperm(length(CL))';

tr = [IM(c(1:200),:) CL(c(1:200),:)];
cand = [IM(c(201:8000),:) CL(c(201:8000),:)];
ts = [IM(c(8001:end),:) CL(c(8001:end),:)];

iterVect = [10:10:200];

% RS
disp('RS');
[tstErrRS, predictionsRS, stdzFinRS, costFinRS] = ...
    AL('RS', tr, cand, ts, iterVect , 10*ones(1,100), num_of_classes);
% AL
disp('AL');
[tstErrAL, predictionsAL, stdzFinAL, costFinAL] = ...
    AL('MS', tr, cand, ts, iterVect, 10*ones(1,100), num_of_classes);

figure
plot(length(tr)+iterVect,tstErrRS(:,1),'-');
hold on
plot(length(tr)+iterVect,tstErrAL(:,1),'r-');
grid on
legend('RS','MS')
xlabel('Samples in training set')
ylabel('Accuracy [pct]')