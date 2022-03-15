%split_scale    Split and (optionally) preprocess data.
% [set1,set2,set3] = split_scale(data,splitRatio,preprocMethod) splits data
% into three non-overlapping sets with ratios of the vector splitRatio and
% preprocesses the sets according to preprocMethod (last column is not
% processed - class attribute).
%
% Parameters:
% data - A nxm matrix.
% splitRatio - A vector of ratios (0 to 1) for all three sets, must add up
%   to 1.
% preprocMethod - Preprocessing method. Choices are:
%   0: No preprocessing
%   1: Normalization to unit hypercube
%   2: Standardization to zero mean - unit variance
%
% Example:
%[trainSet,valSet,testSet] = split_scale(data,[0.6 0.2 0.2],2) splits the
% data into three sets of ratios 0.6, 0.2 and 0.2 and standardizes each set 
% to zero mean and unit variance.

function [set1,set2,set3] = split_scale(data,splitRatio,preprocMethod)
    if (sum(splitRatio) ~= 1)
        disp('Sum of rations is not 1, aborting...');
        return
    end
    % Randomly permutate the row indexes of the data
    idx=randperm(length(data));
    
    % Assign the indexes of set1
    set1Idx=idx(1:round(length(idx)*splitRatio(1)));
    % Assign the indexes of set2
    cumSumRatio2 = splitRatio(1) + splitRatio(2);
    set2Idx=idx(round(length(idx)*splitRatio(1))+1:round(length(idx)*cumSumRatio2));
    % Assign the indexes of set3
    set3Idx=idx(round(length(idx)*cumSumRatio2)+1:end);
    
    % Split data (minus the last column (class))
    set1=data(set1Idx,1:end-1);
    set2=data(set2Idx,1:end-1);
    set3=data(set3Idx,1:end-1);
    
    % Perform the preprocessing, if wanted
    switch preprocMethod
        case 0  % No preprocessing
            
        case 1  % Normalization to unit hypercube
            xmin=min(set1,[],1);
            xmax=max(set1,[],1);
            set1=(set1-repmat(xmin,[length(set1) 1]))./(repmat(xmax,[length(set1) 1])-repmat(xmin,[length(set1) 1]));
            set2=(set2-repmat(xmin,[length(set2) 1]))./(repmat(xmax,[length(set2) 1])-repmat(xmin,[length(set2) 1]));
            set3=(set3-repmat(xmin,[length(set3) 1]))./(repmat(xmax,[length(set3) 1])-repmat(xmin,[length(set3) 1]));
        case 2  % Standardization to zero mean - unit variance
            mu=mean(data(:,1:end-1));
            sig=std(data(:,1:end-1));
            set1=(set1-repmat(mu,[length(set1) 1]))./repmat(sig,[length(set1) 1]);
            set2=(set2-repmat(mu,[length(set2) 1]))./repmat(sig,[length(set2) 1]);
            set3=(set3-repmat(mu,[length(set3) 1]))./repmat(sig,[length(set3) 1]);
        otherwise
            disp('Error, preprocMethod choice not valid, exiting..')
            return;
    end
    % Add the last column to the normalized features
    set1=[set1 data(set1Idx,end)];
    set2=[set2 data(set2Idx,end)];
    set3=[set3 data(set3Idx,end)];

end