% Konstantinidis Konstantinos
% AEM: 9162
% email: konkonstantinidis@ece.auth.gr
close all; 
clear all;
clc;

data = importdata('airfoil_self_noise.dat');

%%% Split data into trainData, valData and testData and stardardize each set
[trainData,checkData,testData] = split_scale(data,[0.6 0.2 0.2],2);

% R-squared metric function (from examples in e-learning)
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%%% Make genfis options 
% (InputMembershipFunctionType is by default bell-shaped)

% TSK_model_1: 2 membership functions, Singleton output type
fis_options(1) = genfisOptions('GridPartition','NumMembershipFunctions',2,...
    'OutputMembershipFunctionType','constant');

% TSK_model_2: 3 membership functions, Singleton output type
fis_options(2) = genfisOptions('GridPartition','NumMembershipFunctions',3,...
    'OutputMembershipFunctionType','constant');

% OutputMembershipFunctionType is by default Polynomial

% TSK_model_3: 2 membership functions, Polynomial output type
fis_options(3) = genfisOptions('GridPartition','NumMembershipFunctions',2);

% TSK_model_4: 3 membership functions, Polynomial output type
fis_options(4) = genfisOptions('GridPartition','NumMembershipFunctions',3);

% Create the models, get the best for parameter combo, generate the
% plots and the prediction errors
% Initialize the models' performance array
modelsPerf = NaN*ones(4,4);
for model=1:4
    disp(['TSK_model_',num2str(model),' training...']);
    
    inFis = genfis(trainData(:,1:end-1),trainData(:,end),fis_options(model));
    
    anfis_options = anfisOptions('InitialFis',inFis,'EpochNumber',100,...
        'ValidationData',checkData,'DisplayANFISInformation',0,...
        'DisplayErrorValues',0,'DisplayStepSize',0);
    
    [trainFis,trainRMSE,stepSize,bestValFis,valRMSE] = anfis(trainData,anfis_options);
    
    %%% Make a figure for membership functions of the features
    figure('Position',[25 70 1500 690]);
    % We have one membership function plot for each feature (5 total)
    % Totalling 10 plots, 5 for the initial model, 5 for the best one
    for f=1:5
        subplot(2,5,f),plotmf(inFis,'input',f);
        title(['Feature ',num2str(f),' before training of model ',num2str(model)]);
        xlabel(['Input ',num2str(f)]);
        subplot(2,5,f+5),plotmf(bestValFis,'input',f);
        title(['Feature ',num2str(f),' of best trained model ',num2str(model)]);
        xlabel(['Input ',num2str(f)]);
    end
    
    %%% Make a figure for the learning curve of the model
    figure('Position',[25 70 1500 690]);
    subplot(2,1,1),plot(trainRMSE,'red');
    hold on;
    subplot(2,1,1),plot(valRMSE,'green');
    hold off;
    title(['Learning curve of TSK model ',num2str(model)]);
    xlabel('Iteration');
    ylabel('RMSE');
    legend('Training RMSE','Testing RMSE','Location','Best');
    
    %%% Calculate the model's performance metrics using testData
    predictions = evalfis(bestValFis,testData(:,1:end-1));
    MSE = mse(predictions,testData(:,end));
    RMSE = sqrt(MSE);
    R2 = Rsq(predictions,testData(:,end));
    %R2 = 1 - NMSE
    NMSE = 1 - R2;
    NDEI = sqrt(NMSE);
    % Store model's perfomance
    modelsPerf(model,:) = [RMSE NMSE NDEI R2];
    
    %%% Make a histogram of the prediction errors
    predErrors = predictions-testData(:,end);
    predErrorsPercent = 100*predErrors./testData(:,end);
    subplot(2,1,2), histogram(predErrorsPercent);
    title(['Percentual Prediction Errors of TSK model ',num2str(model)]);
    ylabel('Frequency');
    xlabel('Prediction Error (%)');
end
% Print models' perfomance metrics
clc;
modelsPerf = array2table(modelsPerf,'VariableNames',{'RMSE' 'NMSE' 'NDEI' 'R2'},...
    'RowNames',{'TSK_model_1','TSK_model_2','TSK_model_3','TSK_model_4'})
save('workspace_TSK_regression_1');