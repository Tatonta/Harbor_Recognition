datasetFolder = 'F:\Università\secondo anno\primo semestre\Caponnetto\PMC 2021 Xibilia Caponetto\sc5'

imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
classes = categories(imds.Labels);
numClasses = numel(classes)
executionEnvironment = "auto";
numLatentInputs = 100;
numObservationsNew = 12;
idxClass = 6;
Z = randn(1,1,numLatentInputs,numObservationsNew,'single');
T = repmat(single(idxClass),[1 1 1 numObservationsNew]);

dlZ = dlarray(Z,'SSCB');
dlT = dlarray(T,'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZ = gpuArray(dlZ);
    dlT = gpuArray(dlT);
end

dlXGenerated = predict(dlnetGenerator,dlZ,dlT);

figure
I = imtile(extractdata(dlXGenerated));
I = rescale(I);
imshow(I)
title("Class: " + classes(idxClass))