clc, clear all
%alex net,resnet,darknet importer,google net,cnn,tensorflow and keras models
%deepNetworkDesigner

%% Chiamo lo script di preprocessing per preparare il dataset
data_preprocessing;
%% costruisco la rete 
net = resnet18;
inputSize = net.Layers(1).InputSize

labelCount = classes

%% Applico delle tecniche di DataAugmenting per arricchire il dataset, per esempio 
%  capovolgendo l'immagine, ruotandola rispetto l'asse X e l'asse Y in
%  maniera randomica
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
'RandXReflection',true, ...
'RandXTranslation',pixelRange, ...
'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imds_train_resampled, 'DataAugmentation', imageAugmenter, 'ColorPreprocessing','gray2rgb');
%% Imposto le opzioni di training per la rete
% sgdm : Sta per stochastic gradient descent with momentum, ovvero
% cerchiamo di scendere nel gradiente per trovare il minimo in una maniera
% randomica per evitare di incastrarci su un minimo locale
% InitialLearnRate : Per scendere il gradiente una delle tecniche è di
% avere un learning rate iniziale molto grade, in maniera tale di evitare
% di rimanere bloccati in un minimo locale
% MaxEpochs e minibatch: Le epoche rappresentano il numero di "forward
% pass" dell'intero dataset. Poichè non è sempre possibile inserire tutto
% il dataset in memoria, è necessario dividere il dataset in gruppi che
% vengono dati in input alla rete, generando un output. Questo approccio
% con i batch inoltre aumenta la generalizzazione della rete.
% L2Regularitazion : Serve a "peggiorare" le prestazioni delle reti, il che
% è fortemente consigliato per reti convoluzionali molto profonde poichè
% tendono a fare overfitting del dataset molto facilmente
% Il Momentum serve ad avere una media pesata per aggiornare i pesi
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 10, 'MiniBatchSize', 64,...
'L2Regularization', 0.0005, 'Momentum', 0.9,...
'Plots','training-progress');

%% analyzeNetwork(net)
numClasses = numel(categories(imds_train_resampled.Labels));
lgraph = layerGraph(net);
% Poichè stiamo applicando transfer learning, cambiamo il numero di classi
% nel fully connected layer in input, in quanto ResNet è stata trainata su un
% dataset di 15 milioni di foto ad alta risoluzione di ImageNet con circa
% 1000 labels
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
% Per avere gli output delle classi che ci interessano, modifichiamo il
% classification layer con le nostre classi
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

% Il comando di training mette in atto il ciclo di "forward pass, backward
% pass (ovvero gradiente avendo la loss) e aggiornamento pesi. 
trainedNet = trainNetwork(augimdsTrain,lgraph,opts);

%validazione
augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages,'ColorPreprocessing','gray2rgb');
predictedLabels = classify(trainedNet,augimdsValidation);
accuracy = mean(predictedLabels == testImages.Labels)

cm = confusionchart(testImages.Labels, predictedLabels,'Normalization','row-normalized')