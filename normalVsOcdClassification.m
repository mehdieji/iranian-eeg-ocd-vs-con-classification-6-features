%%%%%%%%%%%%%%%%%%%%%% Performing the classification %%%%%%%%%%%%%%%%%%%%%%
clc
clear variables
for myRng = 1:25
myRng
clearvars -except myRng
% Parameters
mainData = 1; % The data which is used for training and feature selection
indipendentData = 2; % The independent data which is used as test data
kFold = 8;
% Two different datasets are used in this study. It is specified which
% dataset is used as the main data and which is as the validation data.
dataDirs = {'\Data_1\','\Data_2\'};
dataNames = {'_data1','_data2'};
% Specifying the folder that contains the current code, as the main
% directory and adding this directory to path
mainDir = pwd;
addpath(mainDir);
resultsDir = horzcat(mainDir,'\Results');
% Dir for demographic data
mainDemographicDir = horzcat(mainDir,'\Demographics',dataDirs{mainData});
cd(mainDemographicDir);
load("Info.mat");
ageVector = [Age_Normal;Age_OCD];
% A list of the used features in this study
featuresList = {...
    'Abs_Pow',...
    'Entropy',...
    'HFD',...
    'PLV',...
    'Rel_Pow',...
    'wPLI'};
% There were 2 z-score normalization methods used for this study:
% Median-Iqr (MIZ) and Mean-Std (MSZ)
normalizingMethods = {...
    'MIZ',...
    'MSZ'};
% Creating an empty cell for storing the trained models and their
% performance metrics
mainDataRes = struct();
counter = 0;
% Iterating over all normalization methods
for iNormalizingMethod = 1:length(normalizingMethods)
    iNormalizingMethod
    % Specifying the folder that contains the p-values of the features
    % normalized with the specified z-score normalizing method
    pValueDir = horzcat(...
        mainDir,'\P_Values',dataDirs{mainData},normalizingMethods{iNormalizingMethod});
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirNormal = horzcat(...
        mainDir,'\Features',dataDirs{mainData},normalizingMethods{iNormalizingMethod},'\Normal');
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirOCD = horzcat(...
        mainDir,'\Features',dataDirs{mainData},normalizingMethods{iNormalizingMethod},'\OCD');
    % Iterating over all features
    for iFeature = 1:length(featuresList)
        iFeature
        % Increment for the counter
        counter = counter + 1;
        % P is the p-value variable (loaded when loading p-values .mat
        % files).
        clear P T subscriptsInd Features
        cd(pValueDir)
        load(featuresList{iFeature});
%         [subscriptsInd{1:ndims(P)}] = ind2sub(size(P), find(P(:)<0.05));
%         nGoodFeatures = size(subscriptsInd{1},1);
        % Finding the features that have p-values smaller than 0.05
        goodFeaturesInd = find(P(:)<0.05);
        % The number of good features
        nGoodFeatures = length(goodFeaturesInd);
        cd(featuresDirNormal);
        load(featuresList{iFeature})
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
        % The section below, reshapes the good features from different 
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
        load(featuresList{iFeature})
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
        % Shuffling the data
        rng(myRng);
        idx = randperm(length(y));
        y_numerical = y_numerical(idx);
        X = X(idx,:);
        % Also permute a vector of original indices
        originalIdx = 1:length(y); % creating an index vector
        originalIdx = originalIdx(idx); % permuting it with the same idx

        % Calling the functions that perform classification
        [knnPerf,knnMdl,knnPars,knnRoc,knnPredictions] = knnClassifierFitcecoc(X,y_numerical,kFold,myRng);
        [svmPerf,svmMdl,svmPars,svmRoc,svmPredictions] = svmClassifier(X,y_numerical,kFold,myRng);

        % Matching predictions and ground truth back to the original order
        knnPredictedLabelsOriginalOrder = nan(size(knnPredictions));
        knnPredictedLabelsOriginalOrder(originalIdx) = knnPredictions;
        svmPredictedLabelsOriginalOrder = nan(size(svmPredictions));
        svmPredictedLabelsOriginalOrder(originalIdx) = svmPredictions;
        groundTruthOriginalOrder = nan(size(y_numerical));
        groundTruthOriginalOrder(originalIdx) = y_numerical;

        % Creating age vecror
        ageVectorOrdered = nan(size(y_numerical));
        ageVectorOrdered(originalIdx) = ageVector;
        
        % Storing the results
        mainDataRes(counter).('NormalizationMethod') = ...
            normalizingMethods{iNormalizingMethod};
        mainDataRes(counter).('FeatureType') = ...
            featuresList{iFeature};
        mainDataRes(counter).('GoodFeaturesInd') = ...
            goodFeaturesInd;
        mainDataRes(counter).('knnPerformance') = ...
            knnPerf;
        mainDataRes(counter).('knnModel') = ...
            knnMdl;
        mainDataRes(counter).('knnParameters') = ...
            knnPars;
        mainDataRes(counter).('svmPerformance') = ...
            svmPerf;
        mainDataRes(counter).('svmModel') = ...
            svmMdl;
        mainDataRes(counter).('svmParameters') = ...
            svmPars;
        mainDataRes(counter).('knnRocCurves') = ...
            knnRoc;
        mainDataRes(counter).('svmRocCurves') = ...
            svmRoc;
        mainDataRes(counter).('groundTruthLabels') = ...
            groundTruthOriginalOrder;
        mainDataRes(counter).('knnPredictions') = ...
            knnPredictedLabelsOriginalOrder;
        mainDataRes(counter).('svmPredictions') = ...
            svmPredictedLabelsOriginalOrder;
        mainDataRes(counter).('ageVector') = ...
            ageVectorOrdered;
    end
end
cd(mainDir)
% Testing on independent data
knnRoc = struct();
svmRoc = struct();
independentDataRes = struct();
for iResult = 1:numel(mainDataRes)
    normalizingMethod = mainDataRes(iResult).NormalizationMethod;
    usedFeature = mainDataRes(iResult).FeatureType;
    knnModel = mainDataRes(iResult).knnModel;
    svmModel = mainDataRes(iResult).svmModel;
    goodFeaturesInd = mainDataRes(iResult).GoodFeaturesInd;
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirNormal = horzcat(...
        mainDir,'\Features',dataDirs{indipendentData},normalizingMethod,'\Normal');
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirOCD = horzcat(...
        mainDir,'\Features',dataDirs{indipendentData},normalizingMethod,'\OCD');
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
    [xRocKnn,yRocKnn,~,~] = perfcurve(y_numerical,scoresKnn(:,2),1);
    [xRocSvm,yRocSvm,~,~] = perfcurve(y_numerical,scoresSvm(:,2),1);
    knnRoc(1).('X') = xRocKnn;
    knnRoc(1).('y') = yRocKnn;
    svmRoc(1).('X') = xRocSvm;
    svmRoc(1).('y') = yRocSvm;
    knnPerf = classperf(y_numerical,predsKnn);
    svmPerf = classperf(y_numerical,predsSvm);
    % Storing the results
    independentDataRes(iResult).('NormalizationMethod') = ...
        normalizingMethod;
    independentDataRes(iResult).('FeatureType') = ...
        usedFeature;
    independentDataRes(iResult).('knnPerformance') = ...
        knnPerf;
    independentDataRes(iResult).('svmPerformance') = ...
        svmPerf;
    independentDataRes(iResult).('knnRocCurves') = ...
        knnRoc;
    independentDataRes(iResult).('svmRocCurves') = ...
        svmRoc;
    independentDataRes(iResult).('GroundTruthLabels') = ...
        y_numerical;
end
cd(resultsDir)
save(horzcat('results','_kf',num2str(kFold),'_rng',num2str(myRng)),...
    'mainDataRes','independentDataRes')
cd(mainDir)
end

%% Ploting the Average ROC Curves (Different lines on same plot)
clc
clear variables
close all
% Specifying the folder that contains the current code, as the main
% directory and adding this directory to path
mainDir = pwd;
addpath(mainDir);
resultsDir = horzcat(mainDir,'\Results');
% Parameters
kFold = 8;
% Creating 2 structures for storing the roc curves data for data 1 and 2
rocData1 = struct();
rocData2 = struct();
counter1 = 0;
counter2 = 0;
for myRng = 1:25
clearvars -except myRng mainDir resultsDir kFold rocData1 rocData2 counter1 counter2
matFileName = horzcat('results','_kf',num2str(kFold),'_rng',num2str(myRng));
% Loading the data
cd(resultsDir)
load(matFileName)
for iResult = 1:numel(mainDataRes)
    normalizingMethod = mainDataRes(iResult).NormalizationMethod;
    usedFeature = mainDataRes(iResult).FeatureType;
    for iElement = 1:numel(mainDataRes(iResult).knnRocCurves.TestROC)
        counter1 = counter1 + 1;
        rocData1(counter1).('Normalization') = normalizingMethod;
        rocData1(counter1).('Classifier') = 'KNN';
        rocData1(counter1).('Feature') = usedFeature;
        rocData1(counter1).('X') = mainDataRes(iResult).knnRocCurves.TestROC(iElement).X;
        rocData1(counter1).('Y') = mainDataRes(iResult).knnRocCurves.TestROC(iElement).y;
    end
    for iElement = 1:numel(mainDataRes(iResult).svmRocCurves.TestROC)
        counter1 = counter1 + 1;
        rocData1(counter1).('Normalization') = normalizingMethod;
        rocData1(counter1).('Classifier') = 'SVM';
        rocData1(counter1).('Feature') = usedFeature;
        rocData1(counter1).('X') = mainDataRes(iResult).svmRocCurves.TestROC(iElement).X;
        rocData1(counter1).('Y') = mainDataRes(iResult).svmRocCurves.TestROC(iElement).y;
    end
    counter2 = counter2 + 1;
    rocData2(counter2).('Normalization') = normalizingMethod;
    rocData2(counter2).('Classifier') = 'KNN';
    rocData2(counter2).('Feature') = usedFeature;
    rocData2(counter2).('X') = independentDataRes(iResult).knnRocCurves.X;
    rocData2(counter2).('Y') = independentDataRes(iResult).knnRocCurves.y;
    counter2 = counter2 + 1;
    rocData2(counter2).('Normalization') = normalizingMethod;
    rocData2(counter2).('Classifier') = 'SVM';
    rocData2(counter2).('Feature') = usedFeature;
    rocData2(counter2).('X') = independentDataRes(iResult).svmRocCurves.X;
    rocData2(counter2).('Y') = independentDataRes(iResult).svmRocCurves.y;
end
end
clearvars -except rocData1 rocData2 mainDir
% A list of the used features in this study
featuresList = {...
    'Abs_Pow',...
    'Entropy',...
    'HFD',...
    'PLV',...
    'Rel_Pow',...
    'wPLI'};
% There were 2 z-score normalization methods used for this study:
% Median-Iqr (MIZ) and Mean-Std (MSZ)
normalizingMethods = {...
    'MIZ',...
    'MSZ'};
% Name of the algorithms that we used for classification
classificationAlgorithms = {...
    'KNN',...
    'SVM'};
% Choosing different line styles so that the overlapping ROC curves are
% distinguishable
lineStyles = {'-o','-+','-*','-x','-^','-s'};
% Different colors for plots
colors = {'black','blue','red','green','magenta','cyan'};
% Creating the figures for data 1
figure(1);
ax(1) = gca;
ax(1).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for k-NN Classifier (MSZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
figure(2);
ax(2) = gca;
ax(2).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for SVM Classifier (MSZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
figure(3);
ax(3) = gca;
ax(3).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for k-NN Classifier (MIZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
figure(4);
ax(4) = gca;
ax(4).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for SVM Classifier (MIZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
% Creating the figures for data 2
figure(5);
ax(5) = gca;
ax(5).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for k-NN Classifier (MSZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
figure(6);
ax(6) = gca;
ax(6).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for SVM Classifier (MSZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
figure(7);
ax(7) = gca;
ax(7).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for k-NN Classifier (MIZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
figure(8);
ax(8) = gca;
ax(8).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
hold on
xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
title('\textbf{ROC for SVM Classifier (MIZ Normalization)}', "Interpreter", "latex", "FontSize", 20);
% Plotting the curves for data 1
for iNormalization = 1:numel(normalizingMethods)
    for iClassifier = 1:numel(classificationAlgorithms)
        figure(2*(iNormalization-1)+iClassifier)
        aucArray = zeros(numel(featuresList),1);
        for iFeature = 1:numel(featuresList)
            nSamples = 0;
            goodInd = [];
            uniqueX = [];
            for iSample = 1:numel(rocData1)
                if strcmp(rocData1(iSample).Normalization,normalizingMethods{iNormalization})
                if strcmp(rocData1(iSample).Classifier,classificationAlgorithms{iClassifier})
                if strcmp(rocData1(iSample).Feature,featuresList{iFeature})
                    nSamples = nSamples + 1;
                    goodInd = [goodInd,iSample];
                    uniqueX = [rocData1(iSample).X;uniqueX];
                end
                end
                end
            end
            uniqueX = unique(uniqueX);
            nSamples = length(goodInd);
            nPoints = length(uniqueX);
            meanCurve = zeros(nPoints,1);
            for iSample = 1:nSamples
                ind = goodInd(iSample);
                xData = rocData1(ind).X;
                yData = rocData1(ind).Y;
                tempCurve = zeros(nPoints,1);
                for iPoint = 1:nPoints
                    xTest = uniqueX(iPoint);
                    tempCurve(iPoint) = piecewiseLinInterp(xData,yData,xTest);
                end
                meanCurve = meanCurve + tempCurve/nSamples;
            end
            plot(uniqueX,meanCurve,lineStyles{iFeature},"LineWidth",2,"MarkerSize",7,'Color',colors{iFeature});
            myTicksArray = 0:0.1:1;
            myTicksStringArray = string(myTicksArray) + "    ";
            yticks(myTicksArray);
            yticklabels(myTicksStringArray);
%             plot(uniqueX,meanCurve,'Color',colors{iFeature});
            aucArray(iFeature) = trapz(uniqueX,meanCurve);
        end
        legend(...
            horzcat('AP\hspace{23pt}$AUC: ',num2str(aucArray(1))) + "$",...
            horzcat('AE\hspace{23pt}$AUC: ',num2str(aucArray(2))) + "$",...
            horzcat('HFD\hspace{16pt}$AUC: ',num2str(aucArray(3))) + "$",...
            horzcat('PLV\hspace{18pt}$AUC: ',num2str(aucArray(4)))+ "$",...
            horzcat('RP\hspace{23pt}$AUC: ',num2str(aucArray(5)))+ "$",...
            horzcat('WPLI\hspace{10pt}$AUC: ',num2str(aucArray(6)))+ "$",...
            'Location','southeast',...
            'FontSize',20, "Interpreter", "latex");
    end
end
% Plotting the curves for data 2
for iNormalization = 1:numel(normalizingMethods)
    for iClassifier = 1:numel(classificationAlgorithms)
        figure(4 + 2*(iNormalization-1)+iClassifier)
        aucArray = zeros(numel(featuresList),1);
        for iFeature = 1:numel(featuresList)
            nSamples = 0;
            goodInd = [];
            uniqueX = [];
            for iSample = 1:numel(rocData2)
                if strcmp(rocData2(iSample).Normalization,normalizingMethods{iNormalization})
                if strcmp(rocData2(iSample).Classifier,classificationAlgorithms{iClassifier})
                if strcmp(rocData2(iSample).Feature,featuresList{iFeature})
                    nSamples = nSamples + 1;
                    goodInd = [goodInd,iSample];
                    uniqueX = [rocData2(iSample).X;uniqueX];
                end
                end
                end
            end
            uniqueX = unique(uniqueX);
            nSamples = length(goodInd);
            nPoints = length(uniqueX);
            meanCurve = zeros(nPoints,1);
            for iSample = 1:nSamples
                ind = goodInd(iSample);
                xData = rocData2(ind).X;
                yData = rocData2(ind).Y;
                tempCurve = zeros(nPoints,1);
                for iPoint = 1:nPoints
                    xTest = uniqueX(iPoint);
                    tempCurve(iPoint) = piecewiseLinInterp(xData,yData,xTest);
                end
                meanCurve = meanCurve + tempCurve/nSamples;
            end
            plot(uniqueX,meanCurve,lineStyles{iFeature},"LineWidth",2,"MarkerSize",7,'Color',colors{iFeature});
            myTicksArray = 0:0.1:1;
            myTicksStringArray = string(myTicksArray) + "    ";
            yticks(myTicksArray);
            yticklabels(myTicksStringArray);
%             plot(uniqueX,meanCurve,'Color',colors{iFeature});
            aucArray(iFeature) = trapz(uniqueX,meanCurve);
        end
        legend(...
            horzcat('AP\hspace{23pt}$AUC: ',num2str(aucArray(1))) + "$",...
            horzcat('AE\hspace{23pt}$AUC: ',num2str(aucArray(2))) + "$",...
            horzcat('HFD\hspace{16pt}$AUC: ',num2str(aucArray(3))) + "$",...
            horzcat('PLV\hspace{18pt}$AUC: ',num2str(aucArray(4)))+ "$",...
            horzcat('RP\hspace{23pt}$AUC: ',num2str(aucArray(5)))+ "$",...
            horzcat('WPLI\hspace{10pt}$AUC: ',num2str(aucArray(6)))+ "$",...
            'Location','southeast',...
            'FontSize',20, "Interpreter", "latex");
    end
end
cd(mainDir)


%% Ploting the Average ROC Curves (On subplots + error bars)
clc
clear variables
close all
% Specifying the folder that contains the current code, as the main
% directory and adding this directory to path
mainDir = pwd;
addpath(mainDir);
resultsDir = horzcat(mainDir,'\Results');
% Parameters
kFold = 8;
% Creating 2 structures for storing the roc curves data for data 1 and 2
rocData1 = struct();
rocData2 = struct();
counter1 = 0;
counter2 = 0;
for myRng = 1:25
clearvars -except myRng mainDir resultsDir kFold rocData1 rocData2 counter1 counter2
matFileName = horzcat('results','_kf',num2str(kFold),'_rng',num2str(myRng));
% Loading the data
cd(resultsDir)
load(matFileName)
for iResult = 1:numel(mainDataRes)
    normalizingMethod = mainDataRes(iResult).NormalizationMethod;
    usedFeature = mainDataRes(iResult).FeatureType;
    for iElement = 1:numel(mainDataRes(iResult).knnRocCurves.TestROC)
        counter1 = counter1 + 1;
        rocData1(counter1).('Normalization') = normalizingMethod;
        rocData1(counter1).('Classifier') = 'KNN';
        rocData1(counter1).('Feature') = usedFeature;
        rocData1(counter1).('X') = mainDataRes(iResult).knnRocCurves.TestROC(iElement).X;
        rocData1(counter1).('Y') = mainDataRes(iResult).knnRocCurves.TestROC(iElement).y;
    end
    for iElement = 1:numel(mainDataRes(iResult).svmRocCurves.TestROC)
        counter1 = counter1 + 1;
        rocData1(counter1).('Normalization') = normalizingMethod;
        rocData1(counter1).('Classifier') = 'SVM';
        rocData1(counter1).('Feature') = usedFeature;
        rocData1(counter1).('X') = mainDataRes(iResult).svmRocCurves.TestROC(iElement).X;
        rocData1(counter1).('Y') = mainDataRes(iResult).svmRocCurves.TestROC(iElement).y;
    end
    counter2 = counter2 + 1;
    rocData2(counter2).('Normalization') = normalizingMethod;
    rocData2(counter2).('Classifier') = 'KNN';
    rocData2(counter2).('Feature') = usedFeature;
    rocData2(counter2).('X') = independentDataRes(iResult).knnRocCurves.X;
    rocData2(counter2).('Y') = independentDataRes(iResult).knnRocCurves.y;
    counter2 = counter2 + 1;
    rocData2(counter2).('Normalization') = normalizingMethod;
    rocData2(counter2).('Classifier') = 'SVM';
    rocData2(counter2).('Feature') = usedFeature;
    rocData2(counter2).('X') = independentDataRes(iResult).svmRocCurves.X;
    rocData2(counter2).('Y') = independentDataRes(iResult).svmRocCurves.y;
end
end
clearvars -except rocData1 rocData2 mainDir
% A list of the used features in this study

featuresList = {...
    'Abs_Pow',...
    'Entropy',...
    'PLV',...
    'Rel_Pow',...
    'HFD',...
    'wPLI'};
% A list of the used features in this study

featuresListPlots = {...
    'AP',...
    'AE',...
    'PLV',...
    'RP',...
    'HFD',...
    'wPLI'};
% There were 2 z-score normalization methods used for this study:
% Median-Iqr (MIZ) and Mean-Std (MSZ)
normalizingMethods = {...
    'MIZ',...
    'MSZ'};
% Name of the algorithms that we used for classification
classificationAlgorithms = {...
    'KNN',...
    'SVM'};
% Creating the figures for data 1
figure(1);
ax(1) = gca;
ax(1).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(2);
ax(2) = gca;
ax(2).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(3);
ax(3) = gca;
ax(3).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(4);
ax(4) = gca;
ax(4).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
% Creating the figures for data 2
figure(5);
ax(5) = gca;
ax(5).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(6);
ax(6) = gca;
ax(6).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(7);
ax(7) = gca;
ax(7).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(8);
ax(8) = gca;
ax(8).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);

% Plotting the curves for data 1
for iNormalization = 1:numel(normalizingMethods)
    for iClassifier = 1:numel(classificationAlgorithms)
        figure(2*(iNormalization-1)+iClassifier)
        title_text = "Average ROC Curve for " + classificationAlgorithms{iClassifier} + ...
            " (Primary Test Data) (" + normalizingMethods{iNormalization} + " Normalization)"

        % Add main title using annotation
        annotation('textbox', [0.1, 0.96, 0.8, 0.05], 'String', title_text, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 20, 'FontWeight', 'bold', 'LineStyle', 'none', ...
        'FitBoxToText', 'on', 'FontName', 'Times New Roman');

        aucArray = zeros(numel(featuresList),1);
        for iFeature = 1:numel(featuresList)
            subplot(2,3,iFeature);
            hold on
            nSamples = 0;
            goodInd = [];
            uniqueX = [];
            for iSample = 1:numel(rocData1)
                if strcmp(rocData1(iSample).Normalization,normalizingMethods{iNormalization})
                if strcmp(rocData1(iSample).Classifier,classificationAlgorithms{iClassifier})
                if strcmp(rocData1(iSample).Feature,featuresList{iFeature})
                    nSamples = nSamples + 1;
                    goodInd = [goodInd,iSample];
                    uniqueX = [rocData1(iSample).X;uniqueX];
                end
                end
                end
            end
            uniqueX = unique(uniqueX);
            nSamples = length(goodInd);
            nPoints = length(uniqueX);
            curves = zeros(nSamples, nPoints);
            for iSample = 1:nSamples
                ind = goodInd(iSample);
                xData = rocData1(ind).X;
                yData = rocData1(ind).Y;
                tempCurve = zeros(1,nPoints);
                for iPoint = 1:nPoints
                    xTest = uniqueX(iPoint);
                    tempCurve(iPoint) = piecewiseLinInterp(xData,yData,xTest);
                end
                curves(iSample,:) = tempCurve;
            end
            uniqueX = uniqueX.';
            meanCurve = mean(curves, 1);
            stdCurve = std(curves, 0, 1);
            y_upper = meanCurve + stdCurve;
            y_lower = meanCurve - stdCurve;

            plot(uniqueX, meanCurve, 'b-', 'LineWidth', 2, "MarkerSize",7); % Plot the average data
            fill([uniqueX, fliplr(uniqueX)], [y_lower, fliplr(y_upper)], [0.9 0.9 0.9], 'EdgeColor', 'none'); % Here, [0.9 0.9 0.9] is a light gray color
            plot(uniqueX, meanCurve, 'b-', 'LineWidth', 2, "MarkerSize",7); % Plot the average data
            
            title(featuresListPlots(iFeature));

            xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20, 'FontName', 'Times New Roman');
            ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20, 'FontName', 'Times New Roman');
            
            % X-ticks
            ax = gca;
            ax.XAxis.FontSize = 15;
            ax.XAxis.FontName = 'Times New Roman';

            % Y-ticks
            ylim([0, 1.19]);
            myTicksArray = 0:0.2:1.19;
            myTicksStringArray = string(myTicksArray) + "    ";
            yticks(myTicksArray);
            yticklabels(myTicksStringArray);
            set(gca, 'YTickLabel', myTicksStringArray, 'FontSize', 15, 'FontName', 'Times New Roman');
            
            % Legend for AUC
            aucArray(iFeature) = trapz(uniqueX,meanCurve);
            legend(horzcat('$AUC: ',num2str(aucArray(iFeature))) + "$",...
            'Location','southeast',...
            'FontSize',20, 'FontName', 'Times New Roman', "Interpreter", "latex");

            % Move plots a bit lower
            adjustPosition(-0.02);
        end
    end
end

% Plotting the curves for data 2
for iNormalization = 1:numel(normalizingMethods)
    for iClassifier = 1:numel(classificationAlgorithms)
        figure(4 + 2*(iNormalization-1)+iClassifier)
        title_text = "Average ROC Curve for " + classificationAlgorithms{iClassifier} + ...
            " (Independent Test Data) (" + normalizingMethods{iNormalization} + " Normalization)"

        % Add main title using annotation
        annotation('textbox', [0.1, 0.96, 0.8, 0.05], 'String', title_text, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 20, 'FontWeight', 'bold', 'LineStyle', 'none', ...
        'FitBoxToText', 'on', 'FontName', 'Times New Roman');

        aucArray = zeros(numel(featuresList),1);
        for iFeature = 1:numel(featuresList)
            subplot(2,3,iFeature);
            hold on
            nSamples = 0;
            goodInd = [];
            uniqueX = [];
            for iSample = 1:numel(rocData2)
                if strcmp(rocData2(iSample).Normalization,normalizingMethods{iNormalization})
                if strcmp(rocData2(iSample).Classifier,classificationAlgorithms{iClassifier})
                if strcmp(rocData2(iSample).Feature,featuresList{iFeature})
                    nSamples = nSamples + 1;
                    goodInd = [goodInd,iSample];
                    uniqueX = [rocData2(iSample).X;uniqueX];
                end
                end
                end
            end
            uniqueX = unique(uniqueX);
            nSamples = length(goodInd);
            nPoints = length(uniqueX);
            curves = zeros(nSamples, nPoints);
            for iSample = 1:nSamples
                ind = goodInd(iSample);
                xData = rocData2(ind).X;
                yData = rocData2(ind).Y;
                tempCurve = zeros(1,nPoints);
                for iPoint = 1:nPoints
                    xTest = uniqueX(iPoint);
                    tempCurve(iPoint) = piecewiseLinInterp(xData,yData,xTest);
                end
                curves(iSample,:) = tempCurve;
            end
            uniqueX = uniqueX.';
            meanCurve = mean(curves, 1);
            stdCurve = std(curves, 0, 1);
            y_upper = meanCurve + stdCurve;
            y_lower = meanCurve - stdCurve;

            plot(uniqueX, meanCurve, 'b-', 'LineWidth', 2, "MarkerSize",7); % Plot the average data
            fill([uniqueX, fliplr(uniqueX)], [y_lower, fliplr(y_upper)], [0.9 0.9 0.9], 'EdgeColor', 'none'); % Here, [0.9 0.9 0.9] is a light gray color
            plot(uniqueX, meanCurve, 'b-', 'LineWidth', 2, "MarkerSize",7); % Plot the average data
            
            title(featuresListPlots(iFeature));

            xlabel('False Positive Rate', "Interpreter", "latex", "FontSize", 20);
            ylabel('True Positive Rate', "Interpreter", "latex", "FontSize", 20);
            
            % X-ticks
            ax = gca;
            ax.XAxis.FontSize = 15;
            ax.XAxis.FontName = 'Times New Roman';
            % Y-ticks
            ylim([0, 1.19]);
            myTicksArray = 0:0.2:1.19;
            myTicksStringArray = string(myTicksArray) + "    ";
            yticks(myTicksArray);
            yticklabels(myTicksStringArray);
            set(gca, 'YTickLabel', myTicksStringArray, 'FontSize', 15, 'FontName', 'Times New Roman');
            
            % Legend for AUC
            aucArray(iFeature) = trapz(uniqueX,meanCurve);
            legend(horzcat('$AUC: ',num2str(aucArray(iFeature))) + "$",...
            'Location','southeast',...
            'FontSize',20, "Interpreter", "latex");

            % Move plots a bit lower
            adjustPosition(-0.02);
        end
    end
end

cd(mainDir)

%% Plotting the Confusion Matrices

clc
clear variables
close all

% Parameters
kFold = 8;

% Creating 2 structures for storing the conf mat data for data 1 and 2
cmData1 = struct();
cmData2 = struct();
counter1 = 0;
counter2 = 0;

% A list of the used features in this study
featuresList = {...
    'Abs_Pow',...
    'Entropy',...
    'HFD',...
    'PLV',...
    'Rel_Pow',...
    'wPLI'};

% A list of the used features in this study
featuresListPlots = {...
    'Abs__Pow',...
    'Entropy',...
    'HFD',...
    'PLV',...
    'Rel__Pow',...
    'wPLI'};

% There were 2 z-score normalization methods used for this study:
% Median-Iqr (MIZ) and Mean-Std (MSZ)
normalizingMethods = {...
    'MIZ',...
    'MSZ'};

% Name of the algorithms that we used for classification
classificationAlgorithms = {...
    'KNN',...
    'SVM'};

% Specifying the folder that contains the current code, as the main
% directory and adding this directory to path
mainDir = pwd;
addpath(mainDir);
resultsDir = horzcat(mainDir,'\Results');

counter = 0;
% Looping over the normalization methods and features
for iNormalization = 1:numel(normalizingMethods)
    for iFeature = 1:numel(featuresList)
        normalizingMethod = normalizingMethods{iNormalization};
        usedFeature = featuresList{iFeature};
        knnStruct1 = struct();
        svmStruct1 = struct();
        knnStruct2 = struct();
        svmStruct2 = struct();

        % Looping over different rng results
        for myRng = 1:25
            matFileName = horzcat('results','_kf',num2str(kFold),'_rng',num2str(myRng));
            % Loading the data
            cd(resultsDir)
            load(matFileName)
            % Main data
            for iResult = 1:numel(mainDataRes)
                resNormalizingMethod = mainDataRes(iResult).NormalizationMethod;
                resUsedFeature = mainDataRes(iResult).FeatureType;
                if strcmp(normalizingMethod, resNormalizingMethod) & strcmp(usedFeature, resUsedFeature)
                    knnStruct1(myRng).('KNN') = mainDataRes(iResult).knnPerformance.DiagnosticTable;
                    svmStruct1(myRng).('SVM') = mainDataRes(iResult).svmPerformance.DiagnosticTable;
                end
            end
            % Independent data
            for iResult = 1:numel(independentDataRes)
                resNormalizingMethod = independentDataRes(iResult).NormalizationMethod;
                resUsedFeature = independentDataRes(iResult).FeatureType;
                if strcmp(normalizingMethod, resNormalizingMethod) & strcmp(usedFeature, resUsedFeature)
                    knnStruct2(myRng).('KNN') = independentDataRes(iResult).knnPerformance.DiagnosticTable;
                    svmStruct2(myRng).('SVM') = independentDataRes(iResult).svmPerformance.DiagnosticTable;
                end
            end
        end
        
        counter = counter +1;
        cmData1(counter).('NormalizationMethod') = normalizingMethod;
        cmData1(counter).('FeatureType') = usedFeature;
        cmData1(counter).('KNN') = knnStruct1;
        cmData1(counter).('SVM') = svmStruct1;

        cmData2(counter).('NormalizationMethod') = normalizingMethod;
        cmData2(counter).('FeatureType') = usedFeature;
        cmData2(counter).('KNN') = knnStruct2;
        cmData2(counter).('SVM') = svmStruct2;
    end
end

% Plotting
% Creating the figures for data 1
figure(1);
ax(1) = gca;
ax(1).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(2);
ax(2) = gca;
ax(2).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(3);
ax(3) = gca;
ax(3).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(4);
ax(4) = gca;
ax(4).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
% Creating the figures for data 2
figure(5);
ax(5) = gca;
ax(5).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(6);
ax(6) = gca;
ax(6).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(7);
ax(7) = gca;
ax(7).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);
figure(8);
ax(8) = gca;
ax(8).TickLabelInterpreter = "latex";
set(gcf, 'Position', get(0, 'Screensize'));
set(gca, "FontSize", 18);

% Plotting the confusion matrices for data 1
for iNormalization = 1:numel(normalizingMethods)
    for iClassifier = 1:numel(classificationAlgorithms)
        figure(2*(iNormalization-1)+iClassifier)
        title_text = "Concatenated Confusion Matrices for " + classificationAlgorithms{iClassifier} + ...
            " (Primary Test Data) (" + normalizingMethods{iNormalization} + " Normalization)";

        % Add main title using annotation
        annotation('textbox', [0.1, 0.96, 0.8, 0.05], 'String', title_text, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 20, 'FontWeight', 'bold', 'LineStyle', 'none', ...
        'FitBoxToText', 'on', 'FontName', 'Times New Roman');

        for iFeature = 1:numel(featuresList)
            subplot(2,3,iFeature);
            % Finding the index of the struct corresponding to the
            % normalizing method and feature
            index = ...
                strcmp({cmData1.NormalizationMethod}, normalizingMethods{iNormalization}) & ...
                strcmp({cmData1.FeatureType}, featuresList{iFeature});

            confMat = zeros(2,2);
            for iRng = 1:25
                confMat = confMat + cmData1(index).(classificationAlgorithms{iClassifier})(iRng).(classificationAlgorithms{iClassifier});
            end


            imagesc(confMat);
            colormap('bone');  % Using the "hot" colormap
            cb = colorbar;
            cb.FontSize = 16;
            set(cb, 'FontName', 'Times New Roman');

            % Annotate the heatmap
            cm_values = reshape(confMat, 1, []);
            cm_max = max(cm_values);
            cm_min = min(cm_values);
            cm_med = mean([cm_max, cm_min]);

            if confMat(1, 1) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(1, 1, num2str(confMat(1, 1)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            
            if confMat(2, 1) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(1, 2, num2str(confMat(2, 1)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            
            if confMat(1, 2) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(2, 1, num2str(confMat(1, 2)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            
            if confMat(2, 2) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(2, 2, num2str(confMat(2, 2)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
    
            % Add title and labels
            title(featuresListPlots(iFeature) ,'FontSize', 16, 'FontName', 'Times New Roman');
            xlabel('Actual Class', "Interpreter", "latex", "FontSize", 20, 'FontName', 'Times New Roman');
            ylabel('Predicted Class', "Interpreter", "latex", "FontSize", 20, 'FontName', 'Times New Roman');
            xticks([1, 2]);
            xticklabels({'Positive', 'Negative'});
            yticks([1, 2]);
            yticklabels({'Positive', 'Negative'});
    
            % Adjust axis tick text size
            ax = gca;
            ax.XAxis.FontSize = 16;
            ax.XAxis.FontName = 'Times New Roman';
            ax.YAxis.FontSize = 16;
            ax.YAxis.FontName = 'Times New Roman';
            % Move plots a bit lower
%             adjustPosition(-0.02);
        end
    end
end

% Plotting the confusion matrices for data 2
for iNormalization = 1:numel(normalizingMethods)
    for iClassifier = 1:numel(classificationAlgorithms)
        figure(4 + 2*(iNormalization-1)+iClassifier)
        title_text = "Concatenated Confusion Matrices for " + classificationAlgorithms{iClassifier} + ...
            " (Independent Test Data) (" + normalizingMethods{iNormalization} + " Normalization)";

        % Add main title using annotation
        annotation('textbox', [0.1, 0.96, 0.8, 0.05], 'String', title_text, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 20, 'FontWeight', 'bold', 'LineStyle', 'none', ...
        'FitBoxToText', 'on', 'FontName', 'Times New Roman');

        for iFeature = 1:numel(featuresList)
            subplot(2,3,iFeature);
            % Finding the index of the struct corresponding to the
            % normalizing method and feature
            index = ...
                strcmp({cmData2.NormalizationMethod}, normalizingMethods{iNormalization}) & ...
                strcmp({cmData2.FeatureType}, featuresList{iFeature});

            confMat = zeros(2,2);
            for iRng = 1:25
                confMat = confMat + cmData2(index).(classificationAlgorithms{iClassifier})(iRng).(classificationAlgorithms{iClassifier});
            end

            % Plotting
            imagesc(confMat);
            colormap('bone');  % Using the "hot" colormap
            cb = colorbar;
            cb.FontSize = 16;
            set(cb, 'FontName', 'Times New Roman');

            % Annotate the heatmap
            cm_values = reshape(confMat, 1, []);
            cm_max = max(cm_values);
            cm_min = min(cm_values);
            cm_med = mean([cm_max, cm_min]);

            if confMat(1, 1) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(1, 1, num2str(confMat(1, 1)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            
            if confMat(2, 1) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(1, 2, num2str(confMat(2, 1)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            
            if confMat(1, 2) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(2, 1, num2str(confMat(1, 2)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            
            if confMat(2, 2) > cm_med
                color_code = 'k';
            else
                color_code = 'white';
            end
            text(2, 2, num2str(confMat(2, 2)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 20, ...
                'Color', color_code, 'FontName', 'Times New Roman');
            % Add title and labels
            title(featuresListPlots(iFeature) ,'FontSize', 16, 'FontName', 'Times New Roman');
            xlabel('Actual Class', "Interpreter", "latex", "FontSize", 20, 'FontName', 'Times New Roman');
            ylabel('Predicted Class', "Interpreter", "latex", "FontSize", 20, 'FontName', 'Times New Roman');
            xticks([1, 2]);
            xticklabels({'Positive', 'Negative'});
            yticks([1, 2]);
            yticklabels({'Positive', 'Negative'});
    
            % Adjust axis tick text size
            ax = gca;
            ax.XAxis.FontSize = 16;
            ax.XAxis.FontName = 'Times New Roman';
            ax.YAxis.FontSize = 16;
            ax.YAxis.FontName = 'Times New Roman';
%             % Move plots a bit lower
%             adjustPosition(-0.02);
        end
    end
end

cd(mainDir)

%% The common good features for 2 datasets
mainDir = pwd;
dataDirs = {'\Data_1\','\Data_2\'};
load('optimalClassification_data1_rng2')
res1 = results;
load('optimalClassification_data2_rng2')
res2 = results;
clear results
results = struct();
for iResult = 1:numel(res1)
    clear P T
    normalizingMethod = res1(iResult).NormalizationMethod;
    featureType = res1(iResult).FeatureType;
    goodFeatsInd1 = res1(iResult).GoodFeaturesInd;
    goodFeatsInd2 = res2(iResult).GoodFeaturesInd;
    commonGoodFeatsInd = intersect(goodFeatsInd1,goodFeatsInd2);
    nCommonFeats = length(commonGoodFeatsInd);
    % Specifying the folder that contains the p-values of the features
    % normalized with the specified z-score normalizing method
    pValueDir1 = horzcat(...
        mainDir,'\P_Values',dataDirs{1},normalizingMethod);
    pValueDir2 = horzcat(...
        mainDir,'\P_Values',dataDirs{2},normalizingMethod);
    cd(pValueDir1)
    load(featureType);
    P1 = P(:);
    cd(pValueDir2)
    load(featureType);
    P2 = P(:);
    P_Common = zeros(nCommonFeats,1);
    for iFeature = 1:nCommonFeats
        P_Common(iFeature) = mean(...
            [P1(commonGoodFeatsInd(iFeature)),...
            P2(commonGoodFeatsInd(iFeature))]);
    end
    [~,topFeatsInd] = sort(P_Common(:));
    commonGoodFeatsInd = commonGoodFeatsInd(topFeatsInd);
    results(iResult).('NormalizationMethod') = normalizingMethod;
    results(iResult).('FeatureType') = featureType;
    results(iResult).('GoodFeaturesInd') = commonGoodFeatsInd;
end

%% Publishing Results
% First run the 1st section

for iResult = 1:length(results)
    normalizingMethod = results{iResult}{1}
    usedFeature = results{iResult}{2}
    classifier = 'knn'
    confusion_matrix = results{iResult}{3}.CountingMatrix
    error_rate = results{iResult}{3}.ErrorRate
    sensitivity = results{iResult}{3}.Sensitivity
    specificity = results{iResult}{3}.Specificity
    positive_predictive_value = results{iResult}{3}.PositivePredictiveValue
    negative_predictive_value = results{iResult}{3}.NegativePredictiveValue
    classifier = 'svm'
    confusion_matrix = results{iResult}{4}.CountingMatrix
    error_rate = results{iResult}{4}.ErrorRate
    sensitivity = results{iResult}{4}.Sensitivity
    specificity = results{iResult}{4}.Specificity
    positive_predictive_value = results{iResult}{4}.PositivePredictiveValue
    negative_predictive_value = results{iResult}{4}.NegativePredictiveValue
end

%% Specifying the top 10 features (Forward Feature Selection)
% First run the 1st section of this .m file

% For reproducibility
rng(1)
% Declaring some of the parameters for "sequentialfs" function. c is the
% cross validation parameter. opts is for visualizing the progress of
% feature selection.
c = cvpartition(y,'k',10);
opts = statset('Display','iter');
% Considering an index variable for restoring  data from results variable
% (results from the first section of this .m file)
ind = 0;
% Creating an empty cell for storing the top features of each normalizing
% method, feature, and classification algorithm
results1 = {};
% Iterating over all normalization methods
for iNormalizingMethod = 1:length(normalizingMethods)
    % Specifying the folder that contains the p-values of the features
    % normalized with the specified z-score normalizing method
    pValueDir = horzcat(...
        mainDir,'\P_Values\',normalizingMethods{iNormalizingMethod});
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirNormal = horzcat(...
        mainDir,'\Features\',normalizingMethods{iNormalizingMethod},'\Normal');
    % Specifying the folder that contains the the features of normal
    % subjects, normalized with the specified z-score normalizing method
    featuresDirOCD = horzcat(...
        mainDir,'\Features\',normalizingMethods{iNormalizingMethod},'\OCD');
    % Iterating over all features
    for iFeature = 1:length(featuresList)
        % increment in ind so that we can restore the data saved in the
        % "results" variable (For example we need to restore the optimum
        % number of k in knn)
        ind = ind + 1;
        % P is the p-value variable (loaded when loading p-values .mat
        % files).
        clear P T subscriptsInd Features
        cd(pValueDir)
        load(featuresList{iFeature});
        % Finding the features that have p-values smaller than 0.05
        goodFeaturesInd = find(P(:)<0.05);
        % The number of good features
        nGoodFeatures = length(goodFeaturesInd);
        % Creating an empty array for storing the subscripts of the good
        % features
        goodFeatsSubs = [];
        % Iterating over all of the good features (features with p-value
        % smaller than 0.05)
        for iGoodFeat = 1:nGoodFeatures
            % Since we do not know the number of outputs in ind2sub, we
            % should consider 2 cases. P variable for PLV and wPLI is 3-D
            % and is 2-D for the other 4 features (Could do like what we
            % did using the other dims variable in this code for when we do
            % not know the number of outputs but this was easier!)
            if length(size(P)) == 2
                [row,col] = ...
                    ind2sub(size(P), goodFeaturesInd(iGoodFeat));
                goodFeatsSubs = [goodFeatsSubs;[row,col]];
            elseif length(size(P)) == 3
                [dim1,dim2,dim3] = ...
                    ind2sub(size(P), goodFeaturesInd(iGoodFeat));
                goodFeatsSubs = [goodFeatsSubs;[dim1,dim2,dim3]];
            end
        end
        % Loading the features from normal subjects
        cd(featuresDirNormal);
        load(featuresList{iFeature})
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
        load(featuresList{iFeature})
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
        % Restoring the optimum number of k in knn from the results
        % variable of section 1 of this .m file
        nNeighbors = results{ind}{9};
        % Creating the handle function used in sequencialfs function for
        % the knn model
        funKnn = @(XT,yT,Xt,yt)loss(...
            fitcknn(XT,yT,'NumNeighbors',nNeighbors,'Standardize',1),...
            Xt,yt);
        % Creating the handle function used in sequencialfs function for
        % the knn model
        funSvm = @(XT,yT,Xt,yt)loss(...
            fitcsvm(XT,yT),...
            Xt,yt);
        % Specifying the top 10 features in each classification algorithm
        % for all normalization methods and all features
        [fsKnn,histKnn] = ...
            sequentialfs(funKnn,...
            X,y_numerical,...
            'cv',c,...
            'options',opts,...
            'nfeatures',10);
        [fsSvm,histSvm] = ...
            sequentialfs(funSvm,...
            X,y_numerical,...
            'cv',c,...
            'options',opts,...
            'nfeatures',10);
        % Storing the results for each iteration
        results1{ind} = {...
            goodFeatsSubs,...
            fsKnn,...
            histKnn,...
            fsSvm,...
            histSvm
            };
    end
end
cd(mainDir)
% Name of the 19 electrodes: 
% (1): Fp1, (2): Fp2, (3): F7, (4): F3, (5): C3, (6): C4, (7): Fz, (8): F4
% (9): T3, (10): Cz, (11): T4, (12): T5, (13): P3, (14): P4, (15): T6
% (16): O1, (17): O2, (18): Pz, (19): F8
% We have 18 electrodes in this study (Only No.10 (Cz) not available)

%% Publsihing the feature ranking results (SFS)
% For publishing the results of the above section (copy this script into a
% .m file and use the "publish('___.m','doc')" command

clc
clear variables
load('results')
electrodes = {...
    'Fp1',...
    'Fp2',...
    'F7',...
    'F3',...
    'C3',...
    'C4',...
    'Fz',...
    'F4',...
    'T3',...
    'T4',...
    'T5',...
    'P3',...
    'P4',...
    'T6',...
    'O1',...
    'O2',...
    'Pz',...
    'F8'};
bands = {...
    'Delta',...
    'Theta',...
    'Alpha I',...
    'Alpha II',...
    'Beta I',...
    'Beta II',...
    'Beta III',...
    'Beta IV',...
    'Gamma'};

for iResult = 1:numel(results1)
    normalizingMethod = results{iResult}{1};
    usedFeature = results{iResult}{2};
    [~,indKnn] = sort(sum(results1{iResult}{3}.In));
    indKnn = fliplr(indKnn(end-9:end));
    [~,indSvm] = sort(sum(results1{iResult}{5}.In));
    indSvm = fliplr(indSvm(end-9:end));
    goodFeats = results1{iResult}{1};
    topFeatsKnn = goodFeats(indKnn,:);
    topFeatsSvm = goodFeats(indSvm,:);
    horzcat(...
        normalizingMethod,...
        '_',...
        usedFeature,...
        '_knn')
    for iFeat = 1:size(topFeatsKnn,1)
        if strcmp(usedFeature,'wPLI')||strcmp(usedFeature,'PLV')
            horzcat(...
                electrodes{topFeatsKnn(iFeat,1)},...
                '_',...
                electrodes{topFeatsKnn(iFeat,2)},...
                '_',...
                bands{topFeatsKnn(iFeat,3)})
        else
            horzcat(...
                electrodes{topFeatsKnn(iFeat,1)},...
                '_',...
                bands{topFeatsKnn(iFeat,2)})
        end
    end
    horzcat(...
        normalizingMethod,...
        '_',...
        usedFeature,...
        '_svm')
    for iFeat = 1:size(topFeatsSvm,1)
        if strcmp(usedFeature,'wPLI')|strcmp(usedFeature,'PLV')
            horzcat(...
                electrodes{topFeatsSvm(iFeat,1)},...
                '_',...
                electrodes{topFeatsSvm(iFeat,2)},...
                '_',...
                bands{topFeatsSvm(iFeat,3)})
        else
            horzcat(...
                electrodes{topFeatsSvm(iFeat,1)},...
                '_',...
                bands{topFeatsSvm(iFeat,2)})
        end
    end
end

%% Specifying the top 10 features (P-Values)
% First run the 1st section of this .m file

% Creating an empty cell for storing the top features of each normalizing
% method, feature, and classification algorithm
results1 = {};
% Iterating over all normalization methods
for iNormalizingMethod = 1:length(normalizingMethods)
    % Specifying the folder that contains the p-values of the features
    % normalized with the specified z-score normalizing method
    pValueDir = horzcat(...
        mainDir,'\P_Values\',normalizingMethods{iNormalizingMethod});
    % Iterating over all features
    for iFeature = 1:length(featuresList)
        % increment in ind so that we can restore the data saved in the
        % "results" variable (For example we need to restore the optimum
        % number of k in knn)
        % P is the p-value variable (loaded when loading p-values .mat
        % files).
        clear P T subscriptsInd Features
        cd(pValueDir)
        load(featuresList{iFeature});
        % Finding the features that have p-values smaller than 0.05
        goodFeaturesInd = find(P(:)<0.05);
        % The number of good features
        nGoodFeatures = length(goodFeaturesInd);
        % Creating an empty array for storing the subscripts of the good
        % features
        goodFeatsSubs = [];
        % Iterating over all of the good features (features with p-value
        % smaller than 0.05)
        for iGoodFeat = 1:nGoodFeatures
            % Since we do not know the number of outputs in ind2sub, we
            % should consider 2 cases. P variable for PLV and wPLI is 3-D
            % and is 2-D for the other 4 features (Could do like what we
            % did using the other dims variable in this code for when we do
            % not know the number of outputs but this was easier!)
            if length(size(P)) == 2
                [row,col] = ...
                    ind2sub(size(P), goodFeaturesInd(iGoodFeat));
                goodFeatsSubs = [goodFeatsSubs;[row,col]];
            elseif length(size(P)) == 3
                [dim1,dim2,dim3] = ...
                    ind2sub(size(P), goodFeaturesInd(iGoodFeat));
                goodFeatsSubs = [goodFeatsSubs;[dim1,dim2,dim3]];
            end
        end
        % Specifying the smallest p-values (Best features)
        [~,topFeatsInd] = sort(P(:));
        
        % Specifying the top 10 features in each classification algorithm
        % for all normalization methods and all features
        % Storing the results for each iteration
        results1{end+1} = {...
            topFeatsInd(1:10),...
            goodFeatsSubs,...
            };
    end
end
cd(mainDir)
% Name of the 19 electrodes: 
% (1): Fp1, (2): Fp2, (3): F7, (4): F3, (5): C3, (6): C4, (7): Fz, (8): F4
% (9): T3, (10): Cz, (11): T4, (12): T5, (13): P3, (14): P4, (15): T6
% (16): O1, (17): O2, (18): Pz, (19): F8
% We have 18 electrodes in this study (Only No.10 (Cz) not available)

%% Publsihing the feature ranking results (p-values)
% For publishing the results of the above section (copy this script into a
% .m file and use the "publish('___.m','doc')" command

clc
clear variables
load('results')
electrodes = {...
    'Fp1',...
    'Fp2',...
    'F7',...
    'F3',...
    'C3',...
    'C4',...
    'Fz',...
    'F4',...
    'T3',...
    'T4',...
    'T5',...
    'P3',...
    'P4',...
    'T6',...
    'O1',...
    'O2',...
    'Pz',...
    'F8'};
bands = {...
    'Delta',...
    'Theta',...
    'Alpha I',...
    'Alpha II',...
    'Beta I',...
    'Beta II',...
    'Beta III',...
    'Beta IV',...
    'Gamma'};
for iResult = 1:numel(results1)
    normalizingMethod = results{iResult}{1};
    usedFeature = results{iResult}{2};
    goodFeats = results1{iResult}{1};
    topFeatsInd = results1{iResult}{2};
    topFeats = goodFeats(topFeatsInd,:);
    horzcat(...
        normalizingMethod,...
        '-',...
        usedFeature)
    for iFeat = 1:size(topFeats,1)
        if strcmp(usedFeature,'wPLI')||strcmp(usedFeature,'PLV')
            horzcat(...
                electrodes{topFeats(iFeat,1)},...
                '-',...
                electrodes{topFeats(iFeat,2)},...
                ' (',...
                bands{topFeats(iFeat,3)},...
                ')')
        else
            horzcat(...
                electrodes{topFeats(iFeat,1)},...
                ' (',...
                bands{topFeats(iFeat,2)},...
                ')')
        end
    end
end

%% Used functions


% Knn Classifier
function [perfMetrics,Mdl,bestHypers,rocCurves] = knnClassifier(X,y,kFold,myRng)
    % For reproducibility
    rng(myRng+1)
    % K-NN hyper-parameters:
    % nNeighborsList: The number of neighbors in k-NN
    % distanceTypeList: The defenition for distance in k-NN
    nNeighborsList = ...
        optimizableVariable('nNeigh',[1,30],'Type','integer');
    distanceTypeList = ...
        optimizableVariable('dst',{'chebychev','euclidean','minkowski'},'Type','categorical');
    standardizeList = ...
        optimizableVariable('stan',[0,1],'Type','integer');
    % Creating an object of model performance evaluation with size of
    % the number of samples (when using cross validation, each sample
    % will be a testpoint once so at last the predictions vector will
    % have the same size of y)
    cp = classperf(y);
    % Indexing the datapoints using k-fold
    indices = crossvalind('Kfold',y,kFold);
    % Creating an struct for storing the points of the roc curve in each
    % fold
    rocCurves = struct();
    rocTest = struct();
    rocTrain = struct();
    % iterating over different folds
    for iFold = 1:kFold
        % Splitting test/train data and achieving the indices
        testInd = (indices == iFold); 
        trainInd = ~testInd;
        % Creating a nested cross validation object for selecting the
        % optimal hyper parameters.
        nestedCV = cvpartition(sum(trainInd),'Kfold',kFold);
        % Creating a function for calculating classification loss
        fun = @(x)kfoldLoss(fitcknn(...
            X(trainInd,:),y(trainInd,:),...
            'CVPartition',nestedCV,...
            'NumNeighbors',x.nNeigh,...
            'Distance',char(x.dst),...
            'Standardize',x.stan,...
            'NSMethod','exhaustive'));
        % Specifying the best hyper-parameters in the current iteration of
        % cross validation
        results = bayesopt(...
            fun,[nNeighborsList,distanceTypeList,standardizeList],...
            'Verbose',0,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'PlotFcn',[]);
        bestHypers = bestPoint(results);
        % Training the model using the train data and the best hyper
        % parameters in the current fold
        Mdl = fitcknn(...
            X(trainInd,:),y(trainInd,:),...
            'NumNeighbors',bestHypers.nNeigh,...
            'Distance',char(bestHypers.dst),...
            'Standardize',bestHypers.stan,...
            'NSMethod','exhaustive');
        % Predicting the test data in the current fold using the
        % trained model
        [predictions,scores] = predict(Mdl,X(testInd,:));
        % Creating the roc curves for the test data in each fold
        [xRoc,yRoc,~,~] = perfcurve(y(testInd),scores(:,2),1);
        rocTest(iFold).('X') = xRoc;
        rocTest(iFold).('y') = yRoc;
        % Adding the predictions and groundtruth labels to the object
        % of model performance evaluation with the corresponding
        % indices
        classperf(cp,predictions,testInd);
    end
    % Reporting the metrics of the classifier with the best No. of
    % neighbors
    perfMetrics = cp;
    % Training the optimal model with the whole dataset
    cv = cvpartition(size(X,1),'Kfold',kFold);
    fun = @(x)kfoldLoss(fitcknn(...
        X,y,...
        'CVPartition',cv,...
        'NumNeighbors',x.nNeigh,...
        'Distance',char(x.dst),...
        'Standardize',x.stan,...
        'NSMethod','exhaustive'));
    results = bayesopt(...
            fun,[nNeighborsList,distanceTypeList,standardizeList],...
            'Verbose',0,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'PlotFcn',[]);
    bestHypers = bestPoint(results);
    Mdl = fitcknn(...
        X,y,...
        'NumNeighbors',bestHypers.nNeigh,...
        'Distance',char(bestHypers.dst),...
        'Standardize',bestHypers.stan,...
        'NSMethod','exhaustive');
    [~,scores] = resubPredict(Mdl);
    [xRoc,yRoc,~,~] = perfcurve(y,scores(:,2),1);
    rocTrain(1).('X') = xRoc;
    rocTrain(1).('y') = yRoc;
    rocCurves(1).('TrainingROC') = rocTrain;
    rocCurves(1).('TestROC') = rocTest;
end


% KNN Classifier
function [perfMetrics,Mdl,bestHypers,rocCurves,predictedLabels] = knnClassifierFitcecoc(X,y,kFold,myRng)
    % For reproducibility
    rng(myRng+1)
    % Creating an object of model performance evaluation with size of
    % the number of samples (when using cross validation, each sample
    % will be a testpoint once so at last the predictions vector will
    % have the same size of y)
    cp = classperf(y);
    % Indexing the datapoints using k-fold
    indices = crossvalind('Kfold',y,kFold);
    % Creating an struct for storing the points of the roc curve in each
    % fold
    rocCurves = struct();
    rocTest = struct();
    rocTrain = struct();
    % Initializing the vector to store predicted labels
    predictedLabels = zeros(size(y));
    % iterating over different folds
    for iFold = 1:kFold
        % Splitting test/train data and achieving the indices
        testInd = (indices == iFold); 
        trainInd = ~testInd;
        % Training the model using the train data and the best hyper
        % parameters in the current fold
        Mdl = fitcecoc(...
            X(trainInd,:),y(trainInd,:),...
            'Learners','knn',...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
        % Predicting the test data in the current fold using the
        % trained model
        [predictions,scores] = predict(Mdl,X(testInd,:));
        % Storing predicted labels in the corresponding indices of the vector
        predictedLabels(testInd) = predictions;
        % Creating the roc curves for the test data in each fold
        [xRoc,yRoc,~,~] = perfcurve(y(testInd),scores(:,2),1);
        rocTest(iFold).('X') = xRoc;
        rocTest(iFold).('y') = yRoc;
        % Adding the predictions and groundtruth labels to the object
        % of model performance evaluation with the corresponding
        % indices
        classperf(cp,predictions,testInd);
    end
    % Reporting the metrics of the classifier with the best No. of
    % neighbors
    perfMetrics = cp;
    % Training the optimal model with the whole dataset
    Mdl = fitcecoc(...
        X,y,...
        'Learners','knn',...
        'OptimizeHyperparameters','all',...
        'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
    bestHypers = bestPoint(Mdl.HyperparameterOptimizationResults);
    [~,scores] = resubPredict(Mdl);
    [xRoc,yRoc,~,~] = perfcurve(y,scores(:,2),1);
    rocTrain(1).('X') = xRoc;
    rocTrain(1).('y') = yRoc;
    rocCurves(1).('TrainingROC') = rocTrain;
    rocCurves(1).('TestROC') = rocTest;
end


% SVM Classifier
function [perfMetrics,Mdl,bestHypers,rocCurves,predictedLabels] = svmClassifier(X,y,kFold,myRng)
    % For reproducibility
    rng(myRng+1)
    % Creating an object of model performance evaluation with size of
    % the number of samples (when using cross validation, each sample
    % will be a testpoint once so at last the predictions vector will
    % have the same size of y)
    cp = classperf(y);
    % Indexing the datapoints using k-fold
    indices = crossvalind('Kfold',y,kFold);
    % Creating an struct for storing the points of the roc curve in each
    % fold
    rocCurves = struct();
    rocTest = struct();
    rocTrain = struct();
    % Initializing the vector to store predicted labels
    predictedLabels = zeros(size(y));
    % iterating over different folds
    for iFold = 1:kFold
        % Splitting test/train data and achieving the indices
        testInd = (indices == iFold); 
        trainInd = ~testInd;
        % Training the model using the train data and the best hyper
        % parameters in the current fold
        Mdl = fitcecoc(...
            X(trainInd,:),y(trainInd,:),...
            'Learners','svm',...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
        % Predicting the test data in the current fold using the
        % trained model
        [predictions,scores] = predict(Mdl,X(testInd,:));
        % Storing predicted labels in the corresponding indices of the vector
        predictedLabels(testInd) = predictions;
        % Creating the roc curves for the test data in each fold
        [xRoc,yRoc,~,~] = perfcurve(y(testInd),scores(:,2),1);
        rocTest(iFold).('X') = xRoc;
        rocTest(iFold).('y') = yRoc;
        % Adding the predictions and groundtruth labels to the object
        % of model performance evaluation with the corresponding
        % indices
        classperf(cp,predictions,testInd);
    end
    % Reporting the metrics of the classifier with the best No. of
    % neighbors
    perfMetrics = cp;
    % Training the optimal model with the whole dataset
    Mdl = fitcecoc(...
        X,y,...
        'Learners','svm',...
        'OptimizeHyperparameters','all',...
        'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
    bestHypers = bestPoint(Mdl.HyperparameterOptimizationResults);
    [~,scores] = resubPredict(Mdl);
    [xRoc,yRoc,~,~] = perfcurve(y,scores(:,2),1);
    rocTrain(1).('X') = xRoc;
    rocTrain(1).('y') = yRoc;
    rocCurves(1).('TrainingROC') = rocTrain;
    rocCurves(1).('TestROC') = rocTest;
end


% Interpolate through piecewise linear function
function yInterp = piecewiseLinInterp(xData,yData,xTest)
    equalInd = find(xData==xTest);
    if equalInd
        yInterp = max(yData(equalInd));
    else
        lowerInd = find(xData < xTest, 1, 'last' );
        upperInd = find(xData > xTest, 1 );
        yInterp = interp1(...
            [xData(lowerInd),xData(upperInd)],...
            [yData(lowerInd),yData(upperInd)],...
            xTest,'linear');
    end
end


% Function to adjust the position of the current axis
function adjustPosition(adjustment)
    ax = gca;
    ax.Position(2) = ax.Position(2) + adjustment;
end