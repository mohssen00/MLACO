clc;
clear;

addpath('./files');
addpath('./MLKNN');
addpath('./dataset');

datasets = {'scene', 'yeast', 'Corel5k'};

for d=1:length(datasets)
   
    dataset_name = datasets{d};
    path = strcat('./dataset', dataset_name);
    load(path, 'dataset', 'targets');

    fRange = 10:10:100;
    bucketNum = length(fRange);

    iters = 40;
    % 1.accuracy, 2.HammingLoss, 3.RankingLoss, 4.OneError, 5.Coverage, 6.Average_Precision, 7.time
    aco1_acc = zeros(iters, bucketNum);
    aco1_hamming = zeros(iters, bucketNum);
    aco1_rankingloss = zeros(iters, bucketNum);
    aco1_one_error = zeros(iters, bucketNum);
    aco1_coverage = zeros(iters, bucketNum);
    aco1_average_precision = zeros(iters, bucketNum);
    aco1_time = zeros(1, iters);

    data = cell(iters, 4);
    aco_top = cell(iters, 1);

    decay = 0.1;
    nCycle = 40;
    % pre-proccesing step : remove zero-value feature column
    % zeroF = find(~any(dataset));
    % dataset(:, zeroF) = [];

    for i=1:iters
        disp(i);
        tic;
        nLabels = size(targets, 2);
        [dataTrain, dataTrainLabels, dataTest, dataTestLabels] = splitData(dataset, nLabels);

        nLabels = size(dataTrainLabels, 2);
        [iNum, fNum] = size(dataTrain);

        data{i, 1} = dataTrain;
        data{i, 2} = dataTrainLabels;
        data{i, 3} = dataTest;
        data{i, 4} = dataTestLabels;
        
        start=tic;
        % multi label feature selcetion using ACO
        % O( lnd )
        initialPheromone = abs( 1 - pdist2(dataTrain', dataTrainLabels', 'cosine') );
        % O( ld )
        initialPheromone = max(initialPheromone, [], 2)';
        % O (d)
        initialPheromone = minmax(initialPheromone);

        % calculate features correlation
        % O( nd^2 )
        fCorr = abs (corr(dataTrain, dataTrain));
        fCorr(fCorr == 0) = 0.0001;

        % calculate features-labels correlation
        % O ( lnd )
        flCorr = abs( corr(dataTrain, dataTrainLabels) );
        flCorr = max(flCorr,[], 2)';

        ph1 = MLACO(dataTrain, initialPheromone, nCycle, decay , fCorr, flCorr, 1);
        [ph_val1, ph_idx1] = sort(-ph1);
        aco_top{i} = ph_idx1;
        aco1_time(1, i) = toc(start);

        parfor j=1:bucketNum
            disp(j);

            % 1.accuracy, 2.HammingLoss, 3.RankingLoss, 4.OneError, 5.Coverage, 6.Average_Precision, 7.time
            out1 = MLKNN(dataTrain(:, ph_idx1(1:fRange(j))), dataTrainLabels, dataTest(:, ph_idx1(1:fRange(j))), dataTestLabels);

            aco1_acc(i, j) = out1{1,1};
            aco1_hamming(i, j) = out1{1,2};
            aco1_rankingloss(i, j) = out1{1,3};
            aco1_one_error(i, j) = out1{1,4};
            aco1_coverage(i, j) = out1{1,5};
            aco1_average_precision(i, j) = out1{1,6};

        end

        toc;
    end
    
end
