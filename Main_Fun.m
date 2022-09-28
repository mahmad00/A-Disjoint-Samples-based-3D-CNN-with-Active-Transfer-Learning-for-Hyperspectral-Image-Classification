% Author : Muhammad Ahmad
% Date: 10/09/2022
% Email: mahmad00@gmail.com
% Reference: 3D-CNN-based Active Transfer Learning
%% Required Libraries
% Image Processing for Hyperspectral Imaging
% Machine Learning and Deep Learning
% This demo is executed on Matlab 2021a
%% Clear the workspace and Command Window
clc; clear; close all;
warning('off', 'all');
%% Download KSC Dataset
KSC_url = 'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat';
KSC = 'KSC';
KSC_gt_url = 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat';
KSC_gt = 'KSC_gt';
if ~exist(KSC, 'file') == 1
    fprintf('%s', 'Dataset Downloading:::: ');
    disp(KSC)
    websave(KSC,KSC_url);
end
if ~exist(KSC_gt, 'file') == 1
    websave(KSC_gt,KSC_gt_url);
end
%% Load KSC Dataset
load('KSC_gt');
gt = KSC_gt;
load('KSC');
img = KSC; clear KSC* ans
%% Load Preprocessed Data
load('Settings');
folder = sprintf('%s/KSC', pwd);
if ~exist(folder, 'dir'); mkdir(folder); end
%% Dimensional Reduction and Normalization
[img,~] = hypermnf(img, Dims);
sd = std(img,[],3);
img = img./sd; clear sd
%% Active Learning Methods
for j = 1 : size(AL_Method, 2)
    AL = AL_Method{j};
    switch AL
        case{'Fuz'}
            folder1 = sprintf('%s/Fuz', folder);
            if ~exist(folder1, 'dir'); mkdir(folder1); end
        case{'MI'}
            folder1 = sprintf('%s/MI', folder);
            if ~exist(folder1, 'dir'); mkdir(folder1); end
        case{'BT'}
            folder1 = sprintf('%s/BT', folder);
            if ~exist(folder1, 'dir'); mkdir(folder1); end
    %% End For AL_Methods Switch        
    end
%% Print Active Learning Method
AL_Strtucture = struct('AL', AL, 'M', M, 'h', h);
fprintf('%s', 'Sample Selection Criteria::');
disp(AL_Strtucture.AL);
%% Classification Under Active Learning
[Accuracy, Time] = MY_AL_CNN(img, gt, Tr_Ind, Va_Ind, Te_Ind, AL_Strtucture,...
    Samples, Fuzziness, folder1, WS, Epochs, Class_Names);
%% End For Active Learning Methods 
end