function [predictionV, scoresV, Va_Time, Te_Time, Te1_Time, OAV, kappaV, ...
    AAV, CAV, ConfV, OAT, kappaT, AAT, CAT, ConfT, OAT1, kappaT1, AAT1, CAT1,...
        ConfT1] = Test_Model(net, Va, VaC, Te, TeC, All_Patches, patchLabels,...
            uc, Class_Names, iter, gt, folder, TrainingAccuracy, ...
                ValidationAccuracy, TrainingLoss, ValidationLoss, AL, Va_Ind, Te_Ind)
%% Disjoint Validatation of Trained Network
tic
[predictionV, scoresV] = classify(net,Va);
accuracy = sum(predictionV == VaC)/numel(VaC);
disp(['Accuracy of the validation data = ', num2str(accuracy)])
Va_Time = toc;
%% Dijoint Test of Trained Network
tic
[predictionT, ~] = classify(net,Te);
accuracy = sum(predictionT == TeC)/numel(TeC);
disp(['Accuracy of the Test data = ', num2str(accuracy)])
Te_Time = toc;
%% Complete Test of Trained Network
tic
[predictionT1, ~] = classify(net,All_Patches);
accuracy = sum(predictionT1 == patchLabels)/numel(patchLabels);
disp(['Accuracy of the Test data = ', num2str(accuracy)])
Te1_Time = toc;
%% Disjoint Validation Accuracy
[OAV, kappaV, AAV, CAV, ConfV] = My_Accuracy(double(VaC), ...
        double(predictionV),(1:numel(uc)));
%% Disjoint Test Accuracy
[OAT, kappaT, AAT, CAT, ConfT] = My_Accuracy(double(TeC), ...
    double(predictionT),(1:numel(uc)));    
%% Complete Test Accuracy
[OAT1, kappaT1, AAT1, CAT1, ConfT1] = My_Accuracy(double(patchLabels), ...
    double(predictionT1),(1:numel(uc)));
%% Plot Ground Truths for Predicted Labels (Disjoint/Complete)
% PlotGT(gt, Va_Ind, Te_Ind, double(predictionV)', double(predictionT)',...
%     double(predictionT1)', ConfV, ConfT, ConfT1, Class_Names, folder, iter);
%% Plot Trainnig and Validation Accuracy / Loss
% Plot_Acc_Los(TrainingAccuracy, ValidationAccuracy, ...
%         TrainingLoss, ValidationLoss, AL, folder, iter);
end
%% Internal Function 
%% Computing Accuracy
function [OA, kappa, AA, PC, Conf] = My_Accuracy(True, Predicted, uc)
nrPixelsPerClass = zeros(1,length(uc))';
Conf = zeros(length(uc),length(uc));
for l_true=1:length(uc)
    tmp_true = find (True == l_true);
    nrPixelsPerClass(l_true) = length(tmp_true);
    for l_seg=1:length(uc)
        tmp_seg = find (Predicted == l_seg);
        nrPixels = length(intersect(tmp_true,tmp_seg));
        Conf(l_true,l_seg) = nrPixels;  
    end
end
diagVector = diag(Conf);
PC = (diagVector./(nrPixelsPerClass));
AA = mean(PC);
OA = sum(Predicted == True)/length(True);
kappa = (sum(Conf(:))*sum(diag(Conf)) - sum(Conf)*sum(Conf,2))...
    /(sum(Conf(:))^2 -  sum(Conf)*sum(Conf,2));
end