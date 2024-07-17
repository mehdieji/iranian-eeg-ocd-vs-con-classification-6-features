%% Optimal Hypers
% Normalization method and feature type
% Choosing the normalization/feature combination

iResult = 6;
scoresTrainSVM = zeros(25,1);
scoresTestSVM = zeros(25,1);
scoresTrainKNN = zeros(25,1);
scoresTestKNN = zeros(25,1);

for i=1:25
    load(horzcat('results','_kf8','_rng',num2str(i)))
    % Checking training score for svm
    model = mainDataRes(iResult).svmModel;
    trainPerformance = classperf(model.Y, predict(model, model.X));
    recall = trainPerformance.Sensitivity;
    precision = trainPerformance.PositivePredictiveValue;
    f1socore = (2*recall*precision)/(recall+precision);
    scoresTrainSVM(i) = f1socore;
    % Checking test error for svm
    recall = independentDataRes(iResult).svmPerformance.Sensitivity;
    precision = independentDataRes(iResult).svmPerformance.PositivePredictiveValue;
    f1socore = (2*recall*precision)/(recall+precision);
    scoresTestSVM(i) = f1socore;
    % Checking training score for knn
    model = mainDataRes(iResult).knnModel;
    trainPerformance = classperf(model.Y, predict(model, model.X));
    recall = trainPerformance.Sensitivity;
    precision = trainPerformance.PositivePredictiveValue;
    f1socore = (2*recall*precision)/(recall+precision);
    scoresTrainKNN(i) = f1socore;
    % Checking test score for knn
    recall = independentDataRes(iResult).knnPerformance.Sensitivity;
    precision = independentDataRes(iResult).knnPerformance.PositivePredictiveValue;
    f1socore = (2*recall*precision)/(recall+precision);
    scoresTestKNN(i) = f1socore;
end

[M,I] = max((scoresTrainSVM+scoresTestSVM)/2);
load(horzcat('results','_kf8','_rng',num2str(I)))
mainDataRes(iResult).svmParameters

[M,I] = max((scoresTrainKNN+scoresTestKNN)/2);
load(horzcat('results','_kf8','_rng',num2str(I)))
mainDataRes(iResult).knnParameters

%% Classification Metrics (Data 1)
iResult = 12;

knn_preds = [];
svm_preds = [];
ground_truth = [];

for i=1:25
    load(horzcat('results','_kf8','_rng',num2str(i)))
    % All ages
    ground_truth = [ground_truth; mainDataRes(iResult).groundTruthLabels];
    knn_preds = [knn_preds; mainDataRes(iResult).knnPredictions];
    svm_preds = [svm_preds; mainDataRes(iResult).svmPredictions];
end

knnPerf = classperf(ground_truth, knn_preds);
accKnn = knnPerf.CorrectRate;
senKnn = knnPerf.Sensitivity;
speKnn = knnPerf.Specificity;
ppvKnn = knnPerf.PositivePredictiveValue;
npvKnn = knnPerf.NegativePredictiveValue;

svmPerf = classperf(ground_truth, svm_preds);
accSvm = svmPerf.CorrectRate;
senSvm = svmPerf.Sensitivity;
speSvm = svmPerf.Specificity;
ppvSvm = svmPerf.PositivePredictiveValue;
npvSvm = svmPerf.NegativePredictiveValue;

knnPerf = struct(...
    'acc',round(100*accKnn.',1),...
    'sens',round(100*senKnn.',1),...
    'spec',round(100*speKnn.',1),...
    'ppv',round(100*ppvKnn.',1),...
    'npv',round(100*npvKnn.',1));

svmPerf = struct(...
    'acc',round(100*accSvm.',1),...
    'sens',round(100*senSvm.',1),...
    'spec',round(100*speSvm.',1),...
    'ppv',round(100*ppvSvm.',1),...
    'npv',round(100*npvSvm.',1));

clearvars -except iResult mainDataRes knnPerf svmPerf ground_truth knn_preds svm_preds

%% Classification Metrics (Data 2)
iResult = 12;

knn_cm = zeros(2);
svm_cm = zeros(2);

for i=1:25
    load(horzcat('results','_kf8','_rng',num2str(i)))
    % All ages
    knn_cm = knn_cm + independentDataRes(iResult).knnPerformance.DiagnosticTable;
    svm_cm = svm_cm + independentDataRes(iResult).svmPerformance.DiagnosticTable;
end

% Extract individual values for knn
TP = knn_cm(2,2);
FP = knn_cm(2,1);
FN = knn_cm(1,2);
TN = knn_cm(1,1);
% Construct prediction and ground truth vectors
knn_preds = [ones(1, TP+FP), zeros(1, FN+TN)]';
knn_y = [ones(1, TP), zeros(1, FP+TN), ones(1, FN)]';

% Extract individual values for svm
TP = svm_cm(2,2);
FP = svm_cm(2,1);
FN = svm_cm(1,2);
TN = svm_cm(1,1);
% Construct prediction and ground truth vectors
svm_preds = [ones(1, TP+FP), zeros(1, FN+TN)]';
svm_y = [ones(1, TP), zeros(1, FP+TN), ones(1, FN)]';


knnModelPerf = classperf(knn_y, knn_preds);
accKnn = knnModelPerf.CorrectRate;
senKnn = knnModelPerf.Sensitivity;
speKnn = knnModelPerf.Specificity;
ppvKnn = knnModelPerf.PositivePredictiveValue;
npvKnn = knnModelPerf.NegativePredictiveValue;

svmModelPerf = classperf(svm_y, svm_preds);
accSvm = svmModelPerf.CorrectRate;
senSvm = svmModelPerf.Sensitivity;
speSvm = svmModelPerf.Specificity;
ppvSvm = svmModelPerf.PositivePredictiveValue;
npvSvm = svmModelPerf.NegativePredictiveValue;

knnPerf = struct(...
    'acc',round(100*accKnn.',1),...
    'sens',round(100*senKnn.',1),...
    'spec',round(100*speKnn.',1),...
    'ppv',round(100*ppvKnn.',1),...
    'npv',round(100*npvKnn.',1));

svmPerf = struct(...
    'acc',round(100*accSvm.',1),...
    'sens',round(100*senSvm.',1),...
    'spec',round(100*speSvm.',1),...
    'ppv',round(100*ppvSvm.',1),...
    'npv',round(100*npvSvm.',1));

clearvars -except iResult independentDataRes knn_cm knnModelPerf knnPerf svm_cm svmModelPerf svmPerf ground_truth knn_preds svm_preds

%% Confusion Matrices (Data 1 - Separated by Age)
iResult = 12;

y_all = [];
y_18_30 = [];
y_30_45 = [];
y_45_60 = [];
preds_all_knn = [];
preds_18_30_knn = [];
preds_30_45_knn = [];
preds_45_60_knn = [];
preds_all_svm = [];
preds_18_30_svm = [];
preds_30_45_svm = [];
preds_45_60_svm = [];

for i=1:25
    load(horzcat('results','_kf8','_rng',num2str(i)))
    % All ages
    y = mainDataRes(iResult).groundTruthLabels;
    y_all = [y_all; y];
    % knn
    preds = mainDataRes(iResult).knnPredictions;
    preds_all_knn = [preds_all_knn; preds];
    % svm
    preds = mainDataRes(iResult).svmPredictions;
    preds_all_svm = [preds_all_svm; preds];

    % Aged 18-30
    ind = find(mainDataRes(iResult).ageVector>=18 & mainDataRes(iResult).ageVector<30);
    y = mainDataRes(iResult).groundTruthLabels(ind);
    y_18_30 = [y_18_30; y];
    % knn
    preds = mainDataRes(iResult).knnPredictions(ind);
    preds_18_30_knn = [preds_18_30_knn; preds];
    % svm
    preds = mainDataRes(iResult).svmPredictions(ind);
    preds_18_30_svm = [preds_18_30_svm; preds];
    
    % Aged 30-45
    ind = find(mainDataRes(iResult).ageVector>=30 & mainDataRes(iResult).ageVector<45);
    y = mainDataRes(iResult).groundTruthLabels(ind);
    y_30_45 = [y_30_45; y];
    % knn
    preds = mainDataRes(iResult).knnPredictions(ind);
    preds_30_45_knn = [preds_30_45_knn; preds];
    % svm
    preds = mainDataRes(iResult).svmPredictions(ind);
    preds_30_45_svm = [preds_30_45_svm; preds];

    % Aged 45-60
    ind = find(mainDataRes(iResult).ageVector>=45 & mainDataRes(iResult).ageVector<=60);
    y = mainDataRes(iResult).groundTruthLabels(ind);
    y_45_60 = [y_45_60; y];
    % knn
    preds = mainDataRes(iResult).knnPredictions(ind);
    preds_45_60_knn = [preds_45_60_knn; preds];
    % svm
    preds = mainDataRes(iResult).svmPredictions(ind);
    preds_45_60_svm = [preds_45_60_svm; preds];
end

% Step 1: Compute the confusion matrix
C_all_knn = confusionmat(y_all, preds_all_knn);
C_all_svm = confusionmat(y_all, preds_all_svm);
C_18_30_knn = confusionmat(y_18_30, preds_18_30_knn);
C_18_30_svm = confusionmat(y_18_30, preds_18_30_svm);
C_30_45_knn = confusionmat(y_30_45, preds_30_45_knn);
C_30_45_svm = confusionmat(y_30_45, preds_30_45_svm);
C_45_60_knn = confusionmat(y_45_60, preds_45_60_knn);
C_45_60_svm = confusionmat(y_45_60, preds_45_60_svm);

% Normalized confusion matrix (for percentages)
Cn_all_knn = C_all_knn ./ sum(C_all_knn,2);
Cn_all_svm = C_all_svm ./ sum(C_all_svm,2);
Cn_18_30_knn = C_18_30_knn ./ sum(C_18_30_knn,2);
Cn_18_30_svm = C_18_30_svm ./ sum(C_18_30_svm,2);
Cn_30_45_knn = C_30_45_knn ./ sum(C_30_45_knn,2);
Cn_30_45_svm = C_30_45_svm ./ sum(C_30_45_svm,2);
Cn_45_60_knn = C_45_60_knn ./ sum(C_45_60_knn,2);
Cn_45_60_svm = C_45_60_svm ./ sum(C_45_60_svm,2);

matrices = {C_18_30_knn, C_30_45_knn, C_45_60_knn, C_all_knn,...
    C_18_30_svm, C_30_45_svm, C_45_60_svm, C_all_svm};
normalized_matrices = {Cn_18_30_knn, Cn_30_45_knn, Cn_45_60_knn, Cn_all_knn,...
    Cn_18_30_svm, Cn_30_45_svm, Cn_45_60_svm, Cn_all_svm};

for iMatrix = 1:length(matrices)
    C = matrices{iMatrix};
    C_normalized = normalized_matrices{iMatrix};
    % Step 2: Display the matrices as an image
    figure;
    imagesc(C_normalized); % use normalized version for the color map
    % colorbar;
    colormap('bone'); % or any other colormap you like, e.g., hot, parula, etc.
    caxis([0 1]); % Set the colormap range to [0, 1]
    title('');
    xlabel('');
    ylabel('');
    axis('square');
    
    % Setting the x and y ticks
    classes = {"",""}; % Get unique classes
    numClasses = 2;
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', classes, 'YTick', 1:numClasses, 'YTickLabel', classes);
    
    % Step 3: Add annotations for the actual values and percentages
    for i = 1:numClasses
        for j = 1:numClasses
            if C_normalized(i,j) > 0.5
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(j, i, sprintf('%d\n%.1f%%', C(i,j), 100*C_normalized(i,j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 40, 'Color', color_code);
        end
    end
end

%% Confusion Matrices (Data 2 - Separated by Age)

iResult = 1;

y_all = [];
preds_all_knn = [];
preds_all_svm = [];
age_vector_all = [];


mainData = 1; % The data which is used for training and feature selection
indipendentData = 2; % The independent data which is used as test data
kFold = 8;
% Two different datasets are used in this study. It is specified which
% dataset is used as the main data and which is as the validation data.
dataDirs = {'\Data_1\','\Data_2\'};
dataNames = {'_data1','_data2'};
% Specifying the folder that contains the current code, as the main
% directory and adding this directory to path
resDir = pwd;
parentDir = horzcat(resDir,'\..');
% Dir for demographic data
independentDemographicDir = horzcat(parentDir,'\Demographics',dataDirs{indipendentData});
cd(independentDemographicDir);
load("Info.mat");
ages = [Age_Normal;Age_OCD];

cd(resDir)

for i=1:25
    load(horzcat('results','_kf8','_rng',num2str(i)))

    normalizingMethod = mainDataRes(iResult).NormalizationMethod;
    usedFeature = mainDataRes(iResult).FeatureType;
    knnModel = mainDataRes(iResult).knnModel;
    svmModel = mainDataRes(iResult).svmModel;
    goodFeaturesInd = mainDataRes(iResult).GoodFeaturesInd;
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirNormal = horzcat(...
        parentDir,'\Features',dataDirs{indipendentData},normalizingMethod,'\Normal');
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirOCD = horzcat(...
        parentDir,'\Features',dataDirs{indipendentData},normalizingMethod,'\OCD');
    % The number of good features
    nGoodFeatures = length(goodFeaturesInd);
    cd(featuresDirNormal);
    load(usedFeature)
    % The number of subjects
    nSubjects = size(Features,length(size(Features)));
    % We have 2 classes: OCD, Normal. The classes have the same number
    % of subjects so there will be 2*nSubjects datapoints
    X = zeros(nSubjects*2,nGoodFeatures);
    y = cell(nSubjects*2,1);
    % When the features .mat files are loaded, there will be a variable
    % named "Features" in the workspace. This variable is a 3-D double
    % or a 4-D double based on the loaded feature. One dimension of the
    % variable is for the number of subjects (39 in this study). One
    % other dimension is for the number of bands (9 bands in this
    % study). The other dimension(s) are for the electrodes (18 in this
    % study). For connectivity features (PLV, wPLI) the features are
    % between the electrodes so there will be an additional dimension
    % to the "Features" variable, making it a 4-D double.
    % The section bellow, reshapes the good features from different 
    % bands and electrodes (or cross-electrodes) from each subject,
    % into a row of features (each subject is a datapoint and hence a
    % row). This is done for both OCD and Normal classes.
    otherdims = repmat({':'},1,ndims(Features)-1);
    for iSubject = 1:nSubjects
        subjectFeatures = Features(otherdims{:},iSubject);
        X(iSubject,:) = ...
            subjectFeatures(goodFeaturesInd);
        y{iSubject} = 'Normal';
    end
    cd(featuresDirOCD);
    clear Features
    load(usedFeature)
    for iSubject = 1:nSubjects
        subjectFeatures = Features(otherdims{:},iSubject);
        X(iSubject+nSubjects,:) = ...
            subjectFeatures(goodFeaturesInd);
        y{iSubject+nSubjects} = 'OCD';
    end
    % Converting the classes into binary mode. 1 stands for OCD and 0
    % stands for Normal.
    y_numerical = zeros(length(y),1);
    for iOutput = 1:length(y_numerical)
        if strcmp(y{iOutput},'OCD')
            y_numerical(iOutput) = 1;
        end
    end
    % Calling the functions that perform classification
    [predsKnn,scoresKnn] = predict(knnModel,X);
    [predsSvm,scoresSvm] = predict(svmModel,X);

    y_all = [y_all; y_numerical];
    preds_all_knn = [preds_all_knn; predsKnn];
    preds_all_svm = [preds_all_svm; predsSvm];
    age_vector_all = [age_vector_all; ages];

    cd(resDir)
end


% Aged 18-30
ind = find(age_vector_all>=18 & age_vector_all<30);
y_18_30 = y_all(ind);
% knn
preds_18_30_knn = preds_all_knn(ind);
% svm
preds_18_30_svm = preds_all_svm(ind);

% Aged 30-45
ind = find(age_vector_all>=30 & age_vector_all<45);
y_30_45 = y_all(ind);
% knn
preds_30_45_knn = preds_all_knn(ind);
% svm
preds_30_45_svm = preds_all_svm(ind);

% Aged 45-60
ind = find(age_vector_all>=45 & age_vector_all<=60);
y_45_60 = y_all(ind);
% knn
preds_45_60_knn = preds_all_knn(ind);
% svm
preds_45_60_svm = preds_all_svm(ind);

% Step 1: Compute the confusion matrix
C_all_knn = confusionmat(y_all, preds_all_knn);
C_all_svm = confusionmat(y_all, preds_all_svm);
C_18_30_knn = confusionmat(y_18_30, preds_18_30_knn);
C_18_30_svm = confusionmat(y_18_30, preds_18_30_svm);
C_30_45_knn = confusionmat(y_30_45, preds_30_45_knn);
C_30_45_svm = confusionmat(y_30_45, preds_30_45_svm);
C_45_60_knn = confusionmat(y_45_60, preds_45_60_knn);
C_45_60_svm = confusionmat(y_45_60, preds_45_60_svm);

% Normalized confusion matrix (for percentages)
Cn_all_knn = C_all_knn ./ sum(C_all_knn,2);
Cn_all_svm = C_all_svm ./ sum(C_all_svm,2);
Cn_18_30_knn = C_18_30_knn ./ sum(C_18_30_knn,2);
Cn_18_30_svm = C_18_30_svm ./ sum(C_18_30_svm,2);
Cn_30_45_knn = C_30_45_knn ./ sum(C_30_45_knn,2);
Cn_30_45_svm = C_30_45_svm ./ sum(C_30_45_svm,2);
Cn_45_60_knn = C_45_60_knn ./ sum(C_45_60_knn,2);
Cn_45_60_svm = C_45_60_svm ./ sum(C_45_60_svm,2);

matrices = {C_18_30_knn, C_30_45_knn, C_45_60_knn, C_all_knn,...
    C_18_30_svm, C_30_45_svm, C_45_60_svm, C_all_svm};
normalized_matrices = {Cn_18_30_knn, Cn_30_45_knn, Cn_45_60_knn, Cn_all_knn,...
    Cn_18_30_svm, Cn_30_45_svm, Cn_45_60_svm, Cn_all_svm};

for iMatrix = 1:length(matrices)
    C = matrices{iMatrix};
    C_normalized = normalized_matrices{iMatrix};
    % Step 2: Display the matrices as an image
    figure;
    imagesc(C_normalized); % use normalized version for the color map
    % colorbar;
    colormap('bone'); % or any other colormap you like, e.g., hot, parula, etc.
    caxis([0 1]); % Set the colormap range to [0, 1]
    title('');
    xlabel('');
    ylabel('');
    axis('square');
    
    % Setting the x and y ticks
    classes = {"",""}; % Get unique classes
    numClasses = 2;
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', classes, 'YTick', 1:numClasses, 'YTickLabel', classes);
    
    % Step 3: Add annotations for the actual values and percentages
    for i = 1:numClasses
        for j = 1:numClasses
            if C_normalized(i,j) > 0.5
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(j, i, sprintf('%d\n%.1f%%', C(i,j), 100*C_normalized(i,j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 40, 'Color', color_code);
        end
    end
end

%% Confusion Matrices (Data 2)

iResult = 12;

knn_cm = zeros(2);
svm_cm = zeros(2);

for i=1:25
    load(horzcat('results','_kf8','_rng',num2str(i)))
    % All ages
    knn_cm = knn_cm + independentDataRes(iResult).knnPerformance.DiagnosticTable;
    svm_cm = svm_cm + independentDataRes(iResult).svmPerformance.DiagnosticTable;
end

% Extract individual values for knn
TP = knn_cm(2,2);
FP = knn_cm(2,1);
FN = knn_cm(1,2);
TN = knn_cm(1,1);
% Construct prediction and ground truth vectors
knn_preds = [ones(1, TP+FP), zeros(1, FN+TN)]';
knn_y = [ones(1, TP), zeros(1, FP+TN), ones(1, FN)]';

% Extract individual values for svm
TP = svm_cm(2,2);
FP = svm_cm(2,1);
FN = svm_cm(1,2);
TN = svm_cm(1,1);
% Construct prediction and ground truth vectors
svm_preds = [ones(1, TP+FP), zeros(1, FN+TN)]';
svm_y = [ones(1, TP), zeros(1, FP+TN), ones(1, FN)]';

% Step 1: Compute the confusion matrix
C_all_knn = confusionmat(knn_y, knn_preds);
C_all_svm = confusionmat(svm_y, svm_preds);

% Normalized confusion matrix (for percentages)
Cn_all_knn = C_all_knn ./ sum(C_all_knn,2);
Cn_all_svm = C_all_svm ./ sum(C_all_svm,2);

matrices = {C_all_knn, C_all_svm};
normalized_matrices = {Cn_all_knn, Cn_all_svm};

for iMatrix = 1:length(matrices)
    C = matrices{iMatrix};
    C_normalized = normalized_matrices{iMatrix};
    % Step 2: Display the matrices as an image
    figure;
    imagesc(C_normalized); % use normalized version for the color map
    % colorbar;
    colormap('bone'); % or any other colormap you like, e.g., hot, parula, etc.
    caxis([0 1]); % Set the colormap range to [0, 1]
    title('');
    xlabel('');
    ylabel('');
    axis('square');
    
    % Setting the x and y ticks
    classes = {"",""}; % Get unique classes
    numClasses = 2;
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', classes, 'YTick', 1:numClasses, 'YTickLabel', classes);
    
    % Step 3: Add annotations for the actual values and percentages
    for i = 1:numClasses
        for j = 1:numClasses
            if C_normalized(i,j) > 0.5
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(j, i, sprintf('%d\n%.1f%%', C(i,j), 100*C_normalized(i,j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 40, 'Color', color_code);
        end
    end
end

%% Template Confusion Matrix


% Plotting
% Creating the figures for data 1
figure(1);
ax(1) = gca;
ax(1).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 22);



title_text = "";
% Add main title using annotation
annotation('textbox', [0.1, 0.96, 0.8, 0.05], 'String', title_text, ...
'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
'FontSize', 28, 'FontWeight', 'bold', 'LineStyle', 'none', ...
'FitBoxToText', 'on', 'FontName', 'Times New Roman');



confMat = [100,0;0,100];
imagesc(confMat);
colormap('bone');  % Using the "hot" colormap
caxis([0 100]);
cb = colorbar;
cb.FontSize = 26;
set(cb, 'FontName', 'Times New Roman');
% Add "%" symbol to the ticks
tickValues = get(cb, 'Ticks');
newTickLabels = arrayfun(@(x) sprintf('%.1f%%', x), tickValues, 'UniformOutput', false);
set(cb, 'TickLabels', newTickLabels);

text(1, 1, "True Positives" + newline + newline + "Counts" + newline + "Normalized (%)", 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 26, ...
    'Color', 'k', 'FontName', 'Times New Roman');

text(1, 2, "False Positives" + newline + newline + "Counts" + newline + "Normalized (%)", 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 26, ...
    'Color', 'white', 'FontName', 'Times New Roman');

text(2, 1, "False Negatives" + newline + newline + "Counts" + newline + "Normalized (%)", 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 26, ...
    'Color', 'white', 'FontName', 'Times New Roman');

text(2, 2, "True Negatives" + newline + newline + "Counts" + newline + "Normalized (%)", 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 26, ...
    'Color', 'k', 'FontName', 'Times New Roman');

% Add title and labels
title("Confusion Matrix" ,'FontSize', 28, 'FontName', 'Times New Roman');
xlabel('Predicted Class', "Interpreter", "latex", "FontSize", 26, 'FontName', 'Times New Roman');
ylabel('Actual Class', "Interpreter", "latex", "FontSize", 26, 'FontName', 'Times New Roman');
axis('square');
xticks([1, 2]);
xticklabels({'Positive (OCD)', 'Negative (CON)'});
yticks([1, 2]);
yticklabels({'Positive (OCD)', 'Negative (CON)'});

% Adjust axis tick text size
ax = gca;
ax.XAxis.FontSize = 26;
ax.XAxis.FontName = 'Times New Roman';
ax.YAxis.FontSize = 26;
ax.YAxis.FontName = 'Times New Roman';
% Move plots a bit lower
%             adjustPosition(-0.02);
