clear all, clc
%% Load a pre-trained, deep, convolutional network
imds = imageDatastore('F:\Università\secondo anno\primo semestre\Caponnetto\PMC 2021 Xibilia Caponetto\VAIS_ex', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
net = googlenet;

inputSize = net.Layers(1).InputSize
labelCount =countEachLabel(imds)


%% Modify the network to use five categories
lgraph = layerGraph(net); 

labelCount =countEachLabel(imds)

histogram(imds.Labels)
labels=imds.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);
desiredNumObservationsPerClass = max(numObservations);

files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imds.Files,G);
files = vertcat(files{:});
labels=[];info=strfind(files,'\');
for i=1:numel(files)
    idx=info{i};
    dirName=files{i};
    targetStr=dirName(idx(end-1)+1:idx(end)-1);
    targetStr2=cellstr(targetStr);
    labels=[labels;categorical(targetStr2)];
end

imds.Files = files;
imds.Labels=labels;
labelCount_oversampled = countEachLabel(imds)

[trainingImages, testImages] = splitEachLabel(imds, 0.8, 'randomize');
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
'RandXReflection',true, ...
'RandXTranslation',pixelRange, ...
'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, 'DataAugmentation', imageAugmenter, 'ColorPreprocessing','gray2rgb');


%% Set up our training data
%analyzeNetwork(net)
numClasses = numel(categories(trainingImages.Labels));
lgraph = layerGraph(net);
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
    
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);
inputSize = net.Layers(1).InputSize;
% inputSize= [224 224];

opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 10, 'MiniBatchSize', 64, 'Plots','training-progress',... 
'L2Regularization', 0.0005);

trainedNet = trainNetwork(augimdsTrain,lgraph,opts);

%% validazione
augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages, 'ColorPreprocessing','gray2rgb');
predictedLabels = classify(trainedNet,augimdsValidation);
accuracy = mean(predictedLabels == testImages.Labels)

cm = confusionchart(testImages.Labels, predictedLabels,'Normalization','row-normalized')

