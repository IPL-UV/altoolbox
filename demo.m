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

% Random sampling

disp('SVM with random sampling');

options.model = 'SVM';
options.uncertainty = 'Random';
options.diversity = 'None';
options.iterVect = iterVect;

name = sprintf('%s_%s_%s', options.model, options.uncertainty, options.diversity);
[accCurve.(name) predictions.(name), criterion.(name), sampList.(name), modelParameters.(name)] = ...
             AL(tr, cand, ts, num_of_classes, options);

% AL
disp('SVM with margin sampling active learning');

options.model = 'SVM';
options.uncertainty = 'MS';
options.diversity = 'None';
options.iterVect = iterVect;

name = sprintf('%s_%s_%s', options.model, options.uncertainty, options.diversity);
[accCurve.(name) predictions.(name), criterion.(name), sampList.(name), modelParameters.(name)] = ...
             AL(tr, cand, ts, num_of_classes, options);

figure
plot(length(tr)+iterVect,accCurve.SVM_Random_None(:,1),'r-');
hold on
plot(length(tr)+iterVect,accCurve.SVM_MS_None(:,1),'b-');
grid on
legend('RS','MS')
xlabel('Samples in training set')
ylabel('Accuracy [pct]')
