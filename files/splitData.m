function [dataTrain, dataTrainLabels, dataTest, dataTestLabels] = splitData(dataset, nLabels)
    
instances = dataset(:, 1:end-nLabels);


% instances(:, ~any(instances, 1)) = [];

labels = dataset(:, end-nLabels + 1:end);

% Cross varidation (train: 60%, test: 40%)
cv = cvpartition(size(instances,1),'HoldOut',0.4);
idx = cv.test;

% Separate to training and test data
dataTrain = instances(~idx,:);
dataTest  = instances(idx,:);
dataTrainLabels = labels(~idx, :);
dataTestLabels = labels(idx, :);

% pre-processing
ss=sum(dataTrain);
ff=find(ss==0);
dataTrain(:,ff)=[];
dataTest(:,ff)=[];

ss=sum(dataTrainLabels,2);
ff=find(ss==0);
dataTrain(ff,:)=[];
dataTrainLabels(ff,:)=[];

% zero = ~any(dataTrainLabels);
% dataTrainLabels(:, zero) = [];
% dataTestLabels(:, zero) = [];

end

