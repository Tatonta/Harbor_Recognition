% Copyright 2017 The MathWorks, Inc.
clear all, clc
%% Load a pre-trained, deep, convolutional network
alex = alexnet;
layers = alex.Layers
inputSize = alex.Layers(1).InputSize

%% Modify the network to use five categories
imds = imageDatastore('F:\Università\secondo anno\primo semestre\Caponnetto\PMC 2021 Xibilia Caponetto\sc5', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numClasses = numel(categories(imds.Labels));

layers(23) = fullyConnectedLayer(numClasses);
layers(25) = classificationLayer

%% Set up our training data

labelCount =countEachLabel(imds)

histogram(imds.Labels)
labels=imds.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);
desiredNumObservationsPerClass = max(numObservations);

% files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imds.Files,G);
% files = vertcat(files{:});
% labels=[];info=strfind(files,'\');
% for i=1:numel(files)
%     idx=info{i};
%     dirName=files{i};
%     targetStr=dirName(idx(end-1)+1:idx(end)-1);
%     targetStr2=cellstr(targetStr);
%     labels=[labels;categorical(targetStr2)];
% end

imds.Files = files;
imds.Labels=labels;
labelCount_oversampled = countEachLabel(imds)

[trainingImages, testImages] = splitEachLabel(imds, 0.8, 'randomize');
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
'RandXReflection',true, ...
'RandXTranslation',pixelRange, ...
'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, 'DataAugmentation', imageAugmenter);

% numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end

numBatches = ceil(augimdsTrain.NumObservations / augimdsTrain.MiniBatchSize);
for i = 1:numBatches
    ims = augimdsTrain.read();
    montage(ims{:,1});
    pause;
end
augimds.reset();
%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 10, 'MiniBatchSize', 64, 'Plots','training-progress',... 
'L2Regularization', 0.0005);
myNet = trainNetwork(augimdsTrain, layers, opts);

%% Measure network accuracy

augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages);

predictedLabels = classify(myNet, augimdsValidation); 
accuracy = mean(predictedLabels == testImages.Labels)
cm = confusionchart(testImages.Labels, predictedLabels,'Normalization','row-normalized')

