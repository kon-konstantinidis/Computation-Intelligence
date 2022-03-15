% Konstantinidis Konstantinos
% AEM: 9162
% email: konkonstantinidis@ece.auth.gr
close all;
clear all;
clc;

data = importdata('Superconductivty.csv');
colHeaders = data.colheaders;
data = data.data;

%%% Split data into trainData and testData and stardardize each set
[trainData,testData,~] = split_scale(data,[0.8 0.2 0],2);

% R-squared metric function (from examples in e-learning)
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);


%%%%%%% Search for the optimal model's parameters
numOfFeatures = 1:2:9;
clusterRadius = 0.1:0.1:1;
k_folds = 5;

% Grid Search MSE results for each model
gridMSE = NaN*ones(length(numOfFeatures),length(clusterRadius));
% Number of rules for each model
numRules = NaN*ones(length(numOfFeatures),length(clusterRadius));

% Extract feature importantance via the Relief algorithm (MRMR seemed
% more promising, but I could not easily find it implemented for regression
% in MATLAB, sadly)
[importanceIndexes,importanceWeights] = relieff(trainData(:,1:end-1),trainData(:,end),10);

% Perform a grid search on the parameter numOfFeatures and clusterRadius
% Make a pretty wait bar too
%wb = waitbar(0,'Let''s Go','Name','Progress Bar');
for nF = 1:length(numOfFeatures)
    % Extract the specific number of features before proceeding
    trainData_reduced = trainData(:,importanceIndexes(1:numOfFeatures(nF)));
    % Include target column
    trainData_reduced = [trainData_reduced trainData(:,end)];
    for cR = 1:length(clusterRadius)
        % Display progress
        progress = ((nF-1)*length(clusterRadius)+cR)/(length(numOfFeatures)*length(clusterRadius));
        clc;
        fprintf(['Progress: %3.2f %%, currently at (%d,%d) of %dx%d grid. ',...
            '\nWarning: Progress is linearly calculated, but it''s complexity is exponential. \n'],...
            100*progress,nF,cR,length(numOfFeatures),length(clusterRadius));
        %waitbar(progress,wb,['Currently at [nF,cR]:[',num2str(nF),',',num2str(cR),']']);
        % Initialize cv partition object (i.e. the indices of train and
        % validation set over the iterations of cross validation)
        cvObject = cvpartition(size(trainData,1),'KFold',k_folds);
        
        % MSEs vector for the #k_folds iterations of cross validation
        MSEs = NaN*ones(1,k_folds);
        % Perform k-fold cross validation
        for k=1:k_folds
            % Split the training data into training and validation
            trainDataCV = trainData_reduced(training(cvObject,k),:);
            valDataCV = trainData_reduced(test(cvObject,k),:);
            
            % Proceed to train the model and measure its performance
            fis_options = genfisOptions('SubtractiveClustering',...
                'ClusterInfluenceRange',clusterRadius(cR));
            inFis = genfis(trainDataCV(:,1:end-1),trainDataCV(:,end),fis_options);
            
            % If there haven't been at least two rules generated, anfis
            % will throw an error, so skip ahead, nothing to be done
            if (size(inFis.Rules,2) < 2)
                continue;
            end
            
            anfis_options = anfisOptions('InitialFis',inFis,'EpochNumber',75,...
                'ValidationData',valDataCV,'DisplayANFISInformation',0,...
                'DisplayErrorValues',0,'DisplayStepSize',0,'DisplayFinalResults',0);
            
            [trainFis,trainRMSE,stepSize,bestValFis,valRMSE] = anfis(trainDataCV,anfis_options);
            
            % Store the MSE as this model's performance metric
            MSEs(k) = min(valRMSE.*valRMSE);
        end
        % Store the mean MSE from the %k_folds iterations of CVs
        gridMSE(nF,cR) = mean(MSEs);
        % Store the number of rules
        numRules(nF,cR) = size(inFis.Rules,2);
    end
end
% Having run so far, save workspace
save('workspace_mid_TSK_regression_2');

% Requested Diagrams
% Error relative to number of rules and number of features
figure('Position',[75 70 1400 680]);
% Flatten numRules and gridMSE
numRulesFlat = reshape(numRules,1,[]);
gridMSEFlat = reshape(gridMSE,1,[]);
subplot(1,2,1),scatter(numRulesFlat,gridMSEFlat), hold on;
xlabel('Number of Rules');
ylabel('MSE');
title('Error Relative to Number of Rules');
% Run x-lines for clarity (and style)
lineColors = {'red','green','yellow','magenta'};
rep = 0;
for i=min(numRulesFlat): max(numRulesFlat)
    xline(i,lineColors{mod(rep,4) + 1});
    rep = rep + 1;
end
subplot(1,2,2),boxplot(gridMSE',numOfFeatures);
xlabel('Number of Features');
ylabel('MSE');
title('Error Relative to Number of Features');

%%%% Build the optimal model and metric its performance
% Find the parameter combo with the lowest MSE
[~,bestMSEindexLinear] = min(gridMSE,[],'all','omitnan','linear');
[bestMSErow,bestMSEcol] = ind2sub(size(gridMSE),bestMSEindexLinear);
bestNumOfFeatures = numOfFeatures(bestMSErow);
bestClusterRadius = clusterRadius(bestMSEcol);

%%% Build the model with those parameters

% Extract the best number of features before proceeding
trainData_reduced_best = trainData(:,importanceIndexes(1:bestNumOfFeatures));
testData_reduced_best = testData(:,importanceIndexes(1:bestNumOfFeatures));
% Include target column
trainData_reduced_best = [trainData_reduced_best trainData(:,end)];
testData_reduced_best = [testData_reduced_best testData(:,end)];

% Make yet again a split so we have validation data too
[trainData_reduced_best, valData_reduced_best] = split_scale(trainData_reduced_best,...
    [0.75 0.25 0],0);

% Proceed to train the model and measure its performance
fis_options_best = genfisOptions('SubtractiveClustering',...
    'ClusterInfluenceRange',bestClusterRadius);

inFis_best = genfis(trainData_reduced_best(:,1:end-1),...
    trainData_reduced_best(:,end),fis_options_best);

anfis_options_best = anfisOptions('InitialFis',inFis_best,'EpochNumber',75,...
    'ValidationData',valData_reduced_best,'DisplayANFISInformation',0,...
    'DisplayErrorValues',1,'DisplayStepSize',0,'DisplayFinalResults',1);

[trainFis_best,trainRMSE_best,~,bestValFis_best,valRMSE_best] = ...
    anfis(trainData_reduced_best,anfis_options_best);

% Calculate the best model's performance metrics using testData
predictions = evalfis(bestValFis_best,testData_reduced_best(:,1:end-1));
bestMSE = mse(predictions,testData_reduced_best(:,end));
bestRMSE = sqrt(bestMSE);
bestR2 = Rsq(predictions,testData_reduced_best(:,end));
%R2 = 1 - NMSE
bestNMSE = 1 - bestR2;
bestNDEI = sqrt(bestNMSE);
bestModelPerf = array2table([bestRMSE bestNMSE bestNDEI bestR2],'VariableNames',{'RMSE' 'NMSE' 'NDEI' 'R2'});

%%% Diagrams
% Plot the predictions vs the target values
plotregression(testData_reduced_best(:,end),predictions);
title('Best model''s Predictions vs Target Values');
xlabel('Target Values');
ylabel('Predictions');
saveas(gcf,'Predictions_vs_Target.png');

% Make a histogram of the prediction errors
figure();
predErrors = predictions-testData_reduced_best(:,end);
predErrorsPercent = predErrors./testData_reduced_best(:,end);
histogram(predErrorsPercent,round(length(predErrorsPercent)*15));
title('Percentual Prediction Errors of Best Model');
ylabel('Frequency');
xlabel('Prediction Error (%)');
xlim([-3 10]);
saveas(gcf,'Prediction_Errors_Hist.png');

% Plot the learning curve
figure();
plot(trainRMSE_best,'red');
hold on;
plot(valRMSE_best,'green');
title('Best model''s Error Curve');
xlabel('Iteration');
ylabel('RMSE');
legend('Training RMSE','Validation RMSE','Location','Best');
saveas(gcf,'Learning_Curve.png');

% Show some (3) fuzzy sets start vs finish
figure('Position',[25 50 1500 730]);
subplot(2,3,1),plotmf(inFis_best,'input',4);
subplot(2,3,2),plotmf(inFis_best,'input',6);
subplot(2,3,3),plotmf(inFis_best,'input',9);
subplot(2,3,4),plotmf(bestValFis_best,'input',4);
subplot(2,3,5),plotmf(bestValFis_best,'input',6);
subplot(2,3,6),plotmf(bestValFis_best,'input',9);
saveas(gcf,'Fuzzy_Sets.png');

% Save final workspace
save('workspace_end_TSK_regression_2');