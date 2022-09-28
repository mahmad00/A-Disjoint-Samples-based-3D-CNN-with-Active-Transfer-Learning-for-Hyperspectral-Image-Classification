function [Acc, Time] = MY_AL_CNN(varargin)         
%% Compiling the Input       
%% HSI Cube
img = varargin{1};
if ~numel(img); error('Please Provide HSI Data'); end
%% Ground Truths
gt = varargin{2};
if ~numel(gt); error('Please Provide HSI Ground Truths'); end
%% Training Index
Tr_Ind = varargin{3};
if ~numel(Tr_Ind); error('Please Provide Training Ground Truth Indexes'); end
%% Validation Index
Va_Ind = varargin{4};
if ~numel(Va_Ind); error('Please Provide Validation Ground Truth Indexes'); end
%% Validation Index
Te_Ind = varargin{5};
if ~numel(Te_Ind); error('Please Provide Test Ground Truth Indexes'); end          
%% Active Learning Structure
AL_Strtucture = varargin{6};
if isempty(AL_Strtucture); error('Please Provide Active Learning Parameters');
else; tot_sim = AL_Strtucture.M/AL_Strtucture.h + 1; end
%% Samples Catagory to Select
Samples = varargin{7};
if isempty(Samples); error('Please provide the Sampling Technique'); end
%% Fuzziness Catagory
Fuzziness = varargin{8};
if isempty(Fuzziness); error('Please Provide the Fuzziness Catagorization'); end
%% Folder To Save
folder = varargin{9};
if isempty(folder); error('Please Provide the Directory to Save the Results'); end
%% Training or use Already Trained Model
WS = varargin{10};
if isempty(WS); error('Please Provide patch Size)'); end
%% Epchos
Epochs = varargin{11};
if isempty(Epochs); error('Please Provide the # of Epchos'); end
%% Class Names
Class_Names = varargin{12};
if isempty(Class_Names); error('Please Provide the Name of the Classes'); end
%% Saving Results
Tr_Locations = cell(tot_sim,1); Va_Locations = cell(tot_sim, 1);
Tr_Time = zeros(tot_sim,1); Va_Time = zeros(tot_sim,1); Te_Time = zeros(tot_sim,1);
Te1_Time = zeros(tot_sim,1);
OAV = zeros(tot_sim,1); kappaV = zeros(tot_sim,1); AAV = zeros(tot_sim,1);
CAV = zeros(tot_sim,numel(unique(nonzeros(gt)))); ConfV = cell(tot_sim, 1);
OAT = zeros(tot_sim,1); kappaT = zeros(tot_sim,1); AAT = zeros(tot_sim,1);
CAT = zeros(tot_sim,numel(unique(nonzeros(gt)))); ConfT = cell(tot_sim, 1);
OAT1 = zeros(tot_sim,1); kappaT1 = zeros(tot_sim,1); AAT1 = zeros(tot_sim,1);
CAT1 = zeros(tot_sim,numel(unique(nonzeros(gt)))); ConfT1 = cell(tot_sim, 1);
%% Important Parameters
miniBatchSize = 256;        %% Mini Batch size to process in CNN
initLearningRate = 0.001;   %% Initial Learning Rate
learningRateFactor = 0.01;  %% Learning Rate
%% Patchs
inputSize = [WS WS size(img,3)];
[allPatches,allLabels] = Create_Patches(img,gt,WS);
%% Selecting Non-zero patches
patchesLabeled = allPatches(allLabels>0,:,:,:);     %% Total Patches of HSI
patchLabels = allLabels(allLabels>0);               %% Non Zero Labels
%% Convert One-hot-encoded Labels to Categorical
patchLabels = categorical(patchLabels);
%% Start Active Learning Process Multi Layered ELM
for iter = 1 : tot_sim
    fprintf('Active Selection Round %d \n', iter)
%% Training/Validation/Test Sets
    %% Disjoint Training
    dataInputTrain = patchesLabeled(Tr_Ind,:,:,:);
    dataInputTrain = permute(dataInputTrain,[2 3 4 1]);        %% Training
    TrC = patchLabels(Tr_Ind,1);
    Tr = augmentedImageDatastore(inputSize,dataInputTrain,TrC);
    %% Disjoint Validation
    dataInputVal = patchesLabeled(Va_Ind,:,:,:);
    dataInputVal = permute(dataInputVal,[2 3 4 1]);            %% Validation
    VaC = patchLabels(Va_Ind,1);
    Va = augmentedImageDatastore(inputSize,dataInputVal,VaC);
    %% Disjoint Test
    if iter == 1
        dataInputTest = patchesLabeled(Te_Ind,:,:,:);
        TeC = patchLabels(Te_Ind,1);
        dataInputTest = permute(dataInputTest,[2 3 4 1]);      %% Test
        Te = augmentedImageDatastore(inputSize,dataInputTest,TeC);
    end
    %% Complete HSI and GT for Test
    All_Patches = permute(patchesLabeled,[2 3 4 1]);
    All_Patches = augmentedImageDatastore(inputSize,All_Patches,patchLabels);
    %% 
    TRC = [Tr_Ind' double(TrC)]; VAC = [Va_Ind' double(VaC)];
    Tr_Locations{iter} = TRC; Va_Locations{iter} = VAC;
%% Parameters for CNN
    %% CNN Training Options
        uc = unique(nonzeros(gt));
        layers = [
            image3dInputLayer(inputSize,'Name','Input','Normalization','None')
            convolution3dLayer([5 5 7],60,'Name','conv3d_1')
            reluLayer('Name','Relu_1')
            convolution3dLayer([3 3 5],30,'Name','conv3d_2')
            reluLayer('Name','Relu_2')
            convolution3dLayer([3 3 3],10,'Name','conv3d_3')
            reluLayer('Name','Relu_3')
            fullyConnectedLayer(512,'Name','fc1')
            dropoutLayer(0.4,'Name','drop_1')
            fullyConnectedLayer(256,'Name','fc2')
            dropoutLayer(0.4,'Name','drop_2')
            fullyConnectedLayer(128,'Name','fc3')
            dropoutLayer(0.4,'Name','drop_3')
            fullyConnectedLayer(numel(uc),'Name','fc4')
            softmaxLayer('Name','softmax')
            classificationLayer('Name','output')];
        lgraph = layerGraph(layers);
   %% Specify Training Options
   options = trainingOptions('adam', 'InitialLearnRate', initLearningRate,...
       'LearnRateSchedule','piecewise', 'LearnRateDropPeriod', 10, ...
            'LearnRateDropFactor', learningRateFactor, 'MaxEpochs', Epochs, ...
                'MiniBatchSize', miniBatchSize, 'GradientThresholdMethod',...
                    'l2norm', 'GradientThreshold', 0.01, 'VerboseFrequency'...
                        ,100, 'ValidationData', Va, 'ValidationFrequency',...
                            10, 'ExecutionEnvironment', 'auto');
    %% Train 3D-CNN Model
    tic
    if iter == 1
        [net, info] = trainNetwork(Tr, lgraph, options);
    else
        %% Find the Last Layer
        [learnableLayer, classLayer] = findLayersToReplace(lgraph);
        %% Last Fully Connected Layer
        newLearnableLayer = fullyConnectedLayer(numel(uc), 'Name','fc4');
        lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
        %% Output Layer
        newClassLayer = classificationLayer('Name','output');
        lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
        %% Fine-Tune Network
        [net, info] = trainNetwork(Tr, lgraph, options);
        save(strcat([folder '/', sprintf('net')],".mat"),'net');
    end
    Tr_Time(iter) = toc;
    %% Test the Model and Compute the Accuracies
    [predictionV, scoresV, Va_Time(iter), Te_Time(iter), Te1_Time(iter), ...
        OAV(iter), kappaV(iter), AAV(iter), CAV(iter,:), ConfV{iter},...
            OAT(iter), kappaT(iter), AAT(iter), CAT(iter,:), ConfT{iter},...
                OAT1(iter), kappaT1(iter), AAT1(iter), CAT1(iter,:), ...
                    ConfT1{iter}] = Test_Model(net, Va, VaC, Te, TeC, ...
                        All_Patches, patchLabels, uc, Class_Names, iter,...
                            gt, folder, info.TrainingAccuracy, info.ValidationAccuracy, ...
                                info.TrainingLoss, info.ValidationLoss, ...
                                    AL_Strtucture.AL, Va_Ind, Te_Ind);
    %% Active Learning Sampels Selection    
    if isempty(AL_Strtucture)
        fprintf('No Active Selction')
        %% Fuzziness Based Sample Catagorization
        elseif strcmp(AL_Strtucture.AL, 'Fuz')
            W = My_Member(uc, scoresV);
            Fuz = My_Fuzziness(W);
            Pred1  = [VAC, double(predictionV)];
            Pred = [Fuz, Pred1];
            switch Fuzziness
                case{'High'}
                    [A, ind] = sortrows(Pred, -1);
                case{'Low'}
                    [A, ind] = sortrows(Pred, 1);
            end
            switch Samples
                case{'Misclassified'}
                     [idx, ~] = find(A(:,3) ~= A(:,4));
                case{'Classified'}
                    [idx, ~] = find(A(:,3) == A(:,4));
            end
            index_minME = ind(idx);
            if length(index_minME)>(AL_Strtucture.h)
                xp = index_minME(1 : AL_Strtucture.h)';
            else
                ind(idx) = [];
                index_minME = [index_minME' ind'];
                xp = index_minME(1 : AL_Strtucture.h)';
            end
            TrCNew = VAC(xp,:); TrC = [TRC; TrCNew]; VAC(xp,:) = [];
            %% Select Train and Validation Index's
            Tr_Ind = TrC(:,1)'; Va_Ind = VAC(:,1)';
    %% Mutual Information Based Sample Catagorization
        elseif strcmp(AL_Strtucture.AL, 'MI')
            %% Pick Samples for Membership Matrix
            pactive = sort(scoresV,'descend');
            minME = pactive(:,1) - pactive(:,numel(uc));
            [~, index_minME] = sort(minME);
            xp = index_minME(1 : AL_Strtucture.h);
            TrCNew = VAC(xp,:); TrC = [TRC; TrCNew]; VAC(xp,:) = [];
            %% Select Train and Validation Index's
            Tr_Ind = TrC(:,1)'; Va_Ind = VAC(:,1)';
        %% Breaking Ties Based Sample Catagorization
        elseif strcmp(AL_Strtucture.AL, 'BT')
            %% Pick Samples for Membership Matrix
            pactive = sort(scoresV,'descend');
            minME = pactive(:,1) - pactive(:,2);
            [~, index_minME] = sort(minME);
            xp = index_minME(1 : AL_Strtucture.h);
            TrCNew = VAC(xp,:); TrC = [TRC; TrCNew]; VAC(xp,:) = [];
            %% Select Train and Validation Index's
            Tr_Ind = TrC(:,1)'; Va_Ind = VAC(:,1)';
    %% End for Sample Selection Process
    end
%% End For Iterations on AL.
end
%% Saving Results
%% Accuracy for Disjoint Validation/Test and Complete Test
Acc = [OAV OAT OAT1 AAV AAT AAT1 kappaV kappaT kappaT1];
Accuracy = Acc;
Accuracy = table(Accuracy(:,1), Accuracy(:,2), Accuracy(:,3), Accuracy(:,4),...
    Accuracy(:,5), Accuracy(:,6), Accuracy(:,7), Accuracy(:,8), Accuracy(:,9), ...
        'VariableNames', {'DVa_OA', 'DTe_OA', 'CTe_OA', 'DVa_AA', ...
            'DTe_AA', 'CTe_AA', 'DVa_kappa', 'DTe_kappa', 'CTe_kappa'});
writetable(Accuracy, [folder '/', sprintf('Accuracy.csv')]);
%% Write Table for Disjoinit Validation Per Class Accuracy
Va_Per_Cla = CAV';
Va_PC = table(Va_Per_Cla, 'VariableNames', {'Va_Per_Cla'});
writetable(Va_PC, [folder '/', sprintf('DVal_PC.csv')]);
%% Write Table for Disjoint Test Per Class Accuracy
Te_Per_Cla = CAT';
Te_PC = table(Te_Per_Cla, 'VariableNames', {'Te_Per_Cla'});
writetable(Te_PC, [folder '/', sprintf('DTe_PC.csv')]);
%% Write Table for Complete Test Per Class Accuracy
Te1_Per_Cla = CAT1';
Te1_PC = table(Te1_Per_Cla, 'VariableNames', {'Te_Per_Cla'});
writetable(Te1_PC, [folder '/', sprintf('CTe_PC.csv')]);
%% Write Table for Train, Validation and Test Time
Time = [Tr_Time Va_Time Te_Time Te1_Time];
Time1 = table(Time(:,1), Time(:,2), Time(:,3), Time(:,4), 'VariableNames', ...
    {'DTr', 'DVa', 'DTe', 'CTe'});
writetable(Time1, [folder '/', sprintf('Time.csv')]);
%% Draw Confussion Matrices and Statistical Analysis
% ConfussionMatirx(ConfV, ConfT, ConfT1, Class_Names, folder);
%% Plot Accuracy and Computational Time for Active Learning
%% End for Function
end
%% Internal Functions
%% Fuzzy
function MemberShip = My_Fuzziness(MemberShip)
%% Compute Fuzziness
Fuzziness = zeros(size(MemberShip,1),1);
for l = 1:size(Fuzziness,1)
    Fuzziness(l,:) = fuzziness(MemberShip(l,:));
end
Fuzziness = real(Fuzziness);
MemberShip = nonzeros(Fuzziness);
end
%% Fuzziness 
function E = fuzziness(u_j)
E = 0.01;
c = size(u_j,2);
flt_min = 1.175494e-38;
for i=1:c
    E = E-1/c*(u_j(1,i)*log2(u_j(1,i)+flt_min)+(1-u_j(1,i))*log2(1-u_j(1,i)+flt_min));
end
end
%% Memebership (if Required)
function score = My_Member(uc, score)
%% Reformulate the Membership Matrix as per the defination
for r = 1:size(score,1)
    minVal = 0;
    maxVal = 1/numel(uc);
    [~, ind] = max(score(r,:));
    score(r,:) = minVal + (maxVal - minVal)*rand(1,numel(uc));
    AD1 = sum(score(r,:));
    AB1 = score(r,ind);
    AB2 = AD1 - AB1;
    AB3 = 1 - AB2;
    score(r,ind) = AB3;
end
end
%% Finds the single classification layer and the preceding learnable 
% (fully connected or convolutional) layer of the layer graph lgraph.
function [learnableLayer,classLayer] = findLayersToReplace(lgraph)
if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end
%% Get source, destination, and layer names.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');
%% Find the classification layer. The layer graph must have a single
% classification layer.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);
if sum(isClassificationLayer) ~= 1
    error('Layer graph must have a single classification layer.')
end
classLayer = lgraph.Layers(isClassificationLayer);
%% Traverse the layer graph in reverse starting from the classification
% layer. If the network branches, throw an error.
currentLayerIdx = find(isClassificationLayer);
while true
    if numel(currentLayerIdx) ~= 1
        error('Layer graph must have a single learnable layer preceding the classification layer.')
    end
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
end
end
%% Patches for 3D-CNN
function [patbData,patbLabel] = Create_Patches(HSI, gt, WS)
%% Padding
padding = floor((WS-1)/2);
zeroPaddingPatb = padarray(HSI,[padding,padding],0,'both');
%% 
[r,c,b] = size(HSI);
patbData = zeros(r*c,WS,WS,b);
patbLabel = zeros(r*c,1);
zeroPaddedInput = size(zeroPaddingPatb);
patbIdx = 1;
for i = (padding + 1):(zeroPaddedInput(1) - padding)
    for j = (padding + 1):(zeroPaddedInput(2) - padding)
        patb = zeroPaddingPatb(i - padding:i + padding, j - padding: j + padding,:);
        patbData(patbIdx,:,:,:) = patb;
        patbLabel(patbIdx,1) = gt(i-padding,j-padding);
        patbIdx = patbIdx+1;
    end
end
end