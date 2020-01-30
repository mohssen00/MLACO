function [fitnesses,testPredClass,road_map,Outputs]=ML_knn(road_map,dataSet,labels,zita,Num,Smooth)

[ant_num feature_num]=size(road_map);
%if there is no feature selected,randomly select one feature
% e=sum(road_map,2);
% w=find(e==0);
% for i=1:length(w)
%     s=randperm(feature_num);
%     road_map(w(i),s(1))=1;
% end
%------------compute error rate of reduced data
for i=1:ant_num
    while road_map(i , :) == 0
        road_map(i , :) = round(rand(1 , feature_num));
    end
    
    data2 = dataSet(: , logical(road_map(i , 1:end)));
    train_data=data2(1:round(0.6*size(data2 , 1)) , :);
    train_target=labels(1:round(0.6*size(data2 , 1)) , :);
    test_data=data2(round(0.6*size(data2 , 1))+ 1:end  , :);
    test_target=labels(round(0.6*size(dataSet , 1))+ 1:end  , :);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth); % Invoking the training procedure

    [Outputs(:,:,i),testPredClass(:,:,i)]=MLKNN_test(train_data,train_target,test_data,Num,Prior,PriorN,Cond,CondN); % Performing the test procedure
            predict_label=testPredClass(:,:,i);
%             test_target=test_target';
            [~,~,~,fitnesses(i , 1)] = PrecisionRecall(predict_label,test_target);
% %             fitnesses(i , 1) = zita*sum(sum(testPredClass(:,:,i) == test_target))/length(round(0.6*size(data2 , 1))+ 1:size(dataSet , 1) ) + (1-zita)*(size(road_map,2)-sum(road_map(i , :)))/size(road_map,2);


% % % % % data2 = dataSet;
% % % % %     train_data=data2(1:round(0.6*size(data2 , 1)) , :);
% % % % %     train_target=labels(1:round(0.6*size(data2 , 1)) , :);
% % % % %     test_data=data2(round(0.6*size(data2 , 1))+ 1:end  , :);
% % % % %     test_target=labels(round(0.6*size(dataSet , 1))+ 1:end  , :);
% % % % %     [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth); % Invoking the training procedure
% % % % % 
% % % % %     [Outputs(:,:,i),testPredClass(:,:,i)]=MLKNN_test(train_data,train_target,test_data,Num,Prior,PriorN,Cond,CondN); % Performing the test procedure
% % % % %             predict_label=testPredClass(:,:,i);
% % % % % %             test_target=test_target';
% % % % %             [a,p,r,fitnesses(i , 1)] = PrecisionRecall(predict_label,test_target);


end

end

% % % % % % %     [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,...
% % % % % % %         accuracy,Precision,Recall,F_measure, Outputs(:,:,i),testPredClass(:,:,i)]=...
% % % % % % %         MLKNN_test1(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
% % % % % % 
% % % % % % %     fitnesses(i , 1)=Average_Precision+accuracy+Precision+Recall+F_measure-...
% % % % % % %         HammingLoss-RankingLoss-OneError-Coverage;



    %     [ fitnesses(i,1),predictedLabels(:,i)]=evalute(data2,labels,road_map(i,:),zita);
    %     test_err(i,1)=1-fitnesses(i,1);


