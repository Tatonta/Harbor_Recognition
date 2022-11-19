clear all
%risultato finale dello script: 
% 1 - imageDatastore 1 = imds_train_resampled (con numero di sample per label fissato);
% 2 - imageDatastore 2 = immagini di test con aggiunta di sample provenienti dal dataset di training (sample sovrabbondanti oltre il valore samples_label);
% Nota: le molteplici variabili di appoggio sono rimosse alla fine dello script.

%parametri da selezionare: numero di sample per label, path il dataset, percentuale di split.
samples_label=400;
path='C:\Users\lucar\Desktop\PROCESS MODELING AND CONTROL\PMC_proj\sc5_versione_ridotta\';
split_percentage = 0.8;
%fine parametri selezionabili dall'utente, il resto è automatico.

disp('loading dataset...');
%load del dataset
imds = imageDatastore(path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%figure, histogram(imds.Labels);

%Split del dataset con percentuale fissata
%E' necessario avere "testImages" per poi aggiungervi i sample sovrabbondanti dal dataset di training
[trainingImages, testImages] = splitEachLabel(imds, split_percentage, 'randomize');
%copia di "testImages" ai fini di debug
copia_test_images=testImages;

%NB: gli istogrammi verranno plottati alla fine del processo

%contiamo il numero di labels
label_count_train =countEachLabel(trainingImages);
a=size(label_count_train(:,1));
count_labels=a(:,1);

new_train_locked=imageDatastore({});

%pseudocodice:
% guardo ogni label
% IF (eccedo i bound previsti)
%   THEN {
%          "passo i sample eccedenti a testImages" AND 
%          "li rimuovo dal training" AND
%          "diminuisco la dimensione di "slacciato" a dimensione "samples_label"
%       }
% "appendo "slacciato" a "new_train_images"

disp('first preprocessing: threshold to cut samples, samples to testset...');
%ciclo su ogni label
for k=1:count_labels
    
    %ottengo un imagedatastore che contiene le immagini di una label (ad ogni ciclo) --> slacciato
    ship_label=string(table2array(label_count_train(k,1)));
    %fp contiene il path della subfolder relativa ad uno specifico label (ES: Alilaguna)
    fp=strcat(path,ship_label);
    slacciato = imageDatastore(fp,'LabelSource', 'foldernames');
    [slacciato, unuseful] = splitEachLabel(slacciato, 0.8, 'randomize');
    %calcolo il numero di sample presenti nel label in questione
    dim=size(slacciato.Files);
    dim_r=dim(:,1);
    
    %agiamo solo nel caso in cui il numero dei sample per il label superi il valore desiderato
    if dim_r>samples_label
    %spostiamo i sample sovrabbondanti in testImages
    lower_bound=samples_label+1;
    new_test = imageDatastore(slacciato.Files(lower_bound:dim_r),'LabelSource', 'foldernames');
    temp_test = imageDatastore(cat(1, testImages.Files, new_test.Files)); 
    temp_test.Labels = cat(1, testImages.Labels, new_test.Labels);
    testImages=temp_test;
    
    %per la categoria che ha ecceduto i limiti di samples_labels, contestualmente prendiamo solo i primi N sample (con N pari a samples_label).
    indices = 1:samples_label; 
    slacciato = subset(slacciato,indices); 
    end
    
    %eseguiamo una sorta di append dei sample in "new_train_locked"
    new_test2 = imageDatastore(slacciato.Files,'LabelSource', 'foldernames');
    temp_test2 = imageDatastore(cat(1, new_train_locked.Files, new_test2.Files)); 
    temp_test2.Labels = cat(1, new_train_locked.Labels, new_test2.Labels);
    new_train_locked=temp_test2;
end

disp('first result done: testset created...');
%plot dei risultati
figure, histogram(trainingImages.Labels), title('training images dataset iniziale');
figure,  histogram(new_train_locked.Labels), title('nuovo dataset train');

figure, histogram(copia_test_images.Labels), title('test images dataset iniziale');
figure, histogram(testImages.Labels), title('nuovo dataset test images');

disp('second preprocessing: random resampling...');
%effettuiamo un RESAMPLING RANDOMICO per portare tutti i label ad un numero di campioni pari a "samples_label"
labels=new_train_locked.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);
desiredNumObservationsPerClass = samples_label; %numero di sample per classe

%funzione che itera per randomizzare il resampling
files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},new_train_locked.Files,G);
files = vertcat(files{:});
labels=[];info=strfind(files,'\');
for i=1:numel(files)
    idx=info{i};
    dirName=files{i};
    targetStr=dirName(idx(end-1)+1:idx(end)-1);
    targetStr2=cellstr(targetStr);
    labels=[labels;categorical(targetStr2)];
end

training_images_resampled.Files = files;
training_images_resampled.Labels=labels;

%istogramma dopo del resampling
figure 
histogram(training_images_resampled.Labels)

%conversione da struct a ImageDatastore
%adoperiamo i seguenti dati nel training
%i dati sono già splittati in train e test: i nomi delle variabili sono corrispondenti a quelli usati nel training.
disp('second preprocessing done: trainset created...');
imds.Files = training_images_resampled.Files;
imds.Labels = training_images_resampled.Labels;
imds_train_resampled=imds;

disp('workspace cleaned: END!');
%pulizia del workspace
clear copia_test_images
clear count_labels
clear getta
clear temp_test
clear temp_test2
clear slacciato
clear targetStr
clear targetStr2
clear training_images_resampled
clear ship_label
clear numObservations
clear a k G i
clear copia_test_images
clear dim dim_r
clear dirName
clear new_test new_test2
clear new_train_locked
clear unuseful
clear trainingImages
clear imds
clear label_count_train
clear labelCount
clear indices
clear info labels
clear path idx
clear files
clear fp
clear desiredNumObservationsPerClass
