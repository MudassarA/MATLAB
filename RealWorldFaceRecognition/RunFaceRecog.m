% RunFaceRecog.m
% Main function that reads in files, applies pre-processing, calls face
% recognition algorithm, and then plots resulting accuracy.
%
% Assumptions:
%   - you have a set of pre-processed face images (cropped, normalized, ..)
%   - multiple photographs of each individual to be recognized are
%   available
%   - (maybe?) non-face training data is available?
%
% Alan Brooks and Li Gao
% Northwestern University - ECE 432 Advanced Computer Vision
 
% Mod history:
% 2004:
%    5-May  initial version
%   11-May  added UMIST database capability
%   12-May  prettied the plots, lowered memory requirements
%   13-May  added subfigure, speed up
%   14-May  faster again (caching training data & svd)
%   18-May  began FisherFace technique (Li Gao)
%   19-May  tweaked FisherFace, more work (AB)
%   20-May  tried smaller pictures (AB)
%   25-May  more tries on FisherFace (AB)
%   26-May  added class to ALAN database, smaller UMIST, fix Fisher (AB)
%   31-May  added maxFishFase, figure saving, produced final plots (AB)
%    1-Jun  tried unpreproc ALAN database (AB)

%% ========================================================================
%
%
function RunFaceRecog()
%%
% ------ User Parameters ------
% database type
% dbType = 'UMIST'; % 'ALAN' or 'UMIST' supported so far

% algorithm & feature extraction parameters
frAlgorithm = 'fisher';     % 'eigen' 'fisher' ...?
nEigFace = 21; %21;         % pick # of principal components (PCA)
maxFishEigFace = 41; %Inf;  % pick max number of PCA components to use
                            % in first part of Fisherface algorithm
threshFace = 15; %10; %5;   % thresholds
threshClass = 8; %5; %3;

% caching parameters
train.forceReread = 0;      % 0 = use cached inputs if found, 1 = don't
train.forceNewSvd = 0;      % 0 = use cached SVD, 1 = don't (PCA)
train.forceNewScat = 0;     % 0 = use cached scatters, 1 = don't (LDA)

% plotting parameters
plots.intermediateOn = 1;
plots.finalOn = 1;
plots.savePlotsOn = 1;

% f = filesep;
% switch dbType
% 	case 'UMIST'
%         % data location & type
%         basePath = ['C:\Documents and Settings\alanb\My Documents\NWU\' ...
%                 'ECE 432\final project\face_databases\UMIST'];
%         %basePath = ['D:\My Documents\NWU\ECE 432 Comp Vision\' ...
%         %        'face_databases\UMIST'];
%         %basePath = ['C:\Documents and Settings\lgao\My Documents\' ...
%         %        'ECE432\Projects\Facial Recognition\'];
%         %db = 'croppedConsolPng';
%         %db = 'croppedConsolPng46x56';
%         %db = 'croppedConsolPng23x28';
%         db = 'croppedConsolPng23x28_subset'; % 3 train, 1 recog image
%         imType = 'png';
% 	case 'ALAN'
% 		% data location & type
% 		if isunix
% 			basePath = ['/Users/alanb/Documents/Riting & Labs/' ...
%                     'Northwestern/Year 2 Q3 (Spring)/' ...
%                     'EE 432 Computer Vision/final project/' ...
%                     'face_databases/MUstudentPhotos'];		
%         else
%             basePath = ['C:\Documents and Settings\alanb\My Documents\' ...
%                     'NWU\ECE 432\final project\face_databases\' ...
%                     'MUstudentPhotos'];
%             %basePath = ['D:\My Documents\NWU\ECE 432 Comp Vision\' ...
%             %        'face_databases\millikin'];
% 		end
% 		%db = 'thumbsMoreThan2prepro40x60';
% 		%db = 'thumbsMoreThan2prepro10x15';
% 		%db = 'tmbwClassOval40x60';
% 		db = 'tmbwClassOval40x60_2'; % changed which ones in recog
% 		%db = 'tmbwClassppNo60x80g'; % no pre-processing
% 		imType = 'png';
%     otherwise
%         error(sprintf('%s database type not understood.',dbType))
% end
% 
% train.path = fullfile(basePath,[db f 'train']);
% recog.path = fullfile(basePath,[db f 'recog']);
imType = 'jpg';
train.path = 'projectFaces/A/';
recog.path = 'projectFaces/D/';

% ------ Pre-Processing ------
% read all training images
di = dir(fullfile(train.path,['*.' imType]));
if isempty(di)
    fprintf('Err: Couldn''t find any %s files in training path:\n\t%s\n',...
        imType,train.path)
    return
end

inputTrainFile = fullfile(train.path,'storedInputs.mat');
foundCache = exist(inputTrainFile,'file');
if foundCache & ~train.forceReread 
    % loads stored input images
    trainNow = train;
    load(inputTrainFile)
    trainNow.I = train.I;   % only update desired vars
    trainNow.classNameTrue = train.classNameTrue;    
    trainNow.classTrue = train.classTrue;
    train = trainNow;
    clear trainNow
else
    % read training images
	%info = imfinfo(fullfile(train.path,di(1).name));
	%Nx = info.Height;
	%Ny = info.Width;
    Nx = 70;
    Ny = 70;
	M = length(di); % number of training images
	train.I = uint8(zeros(Nx,Ny,M)); % init for speed
    numOfClass = 0; % initialize the number of classes 
    classBoundary = 0;
    prevIndex = 1;
	for i = 1:M % for each training image
        % Read and store class
        [I,mp] = imread(fullfile(train.path,di(i).name),imType);
        I = I(81:150,81:150,:);
        train.I(:,:,i) = rgb2gray(I);     % image
        %train.I(:,:,i) = adapthisteq(I,'clipLimit',0.001);
        %train.I(:,:,i) = imadjust(I);
        %train.I(:,:,i) = histeq(I);
        train.classNameTrue{i} = di(i).name(1:end-4); % specific for my names
        train.classTrue(i) = i;
        
        % Group & calculate statistics for each class
        %   looks for new classes by 1st 2 letters in filename
        if i>1
            if ~strcmp(di(i-1).name(1:2),di(i).name(1:2)) % class bndry
                j = i;
                classBoundary = 1;
            end
        end
        if i==M % force class bndry for last image
            j = i+1;
            classBoundary = 1;
        end
        if classBoundary
            numOfClass = numOfClass + 1;
            trainClass.className{numOfClass} = ['class_' di(j-1).name(1:2)];
            trainClass.num(numOfClass) = j-prevIndex;
            X = double(reshape(...
                train.I(:,:,prevIndex:(j-1)),[Nx*Ny j-prevIndex]))./256;
            trainClass.mean{numOfClass} = mean(X,2);
            trainClass.classStartIndex(numOfClass) = prevIndex;
            trainClass.classEndIndex(numOfClass) = j-1;
            prevIndex = j;
            clear X;
            classBoundary = 0;
        end
	end
    
    % save cached images for next run
    save(inputTrainFile,'train','Nx','Ny','M','mp','trainClass')
end

% read all images to be recognized
di = dir(fullfile(recog.path,['*.' imType]));
if isempty(di)
    fprintf('Err: Couldn''t find any %s files in recog path:\n\t%s\n',...
        imType,recog.path)
    return
end
% info = imfinfo(fullfile(recog.path,di(1).name));
% if (info.Height ~= Nx) | (info.Width ~= Ny)
%     error('recog images must be same size as training images')
% end
M2 = length(di); % number of recog images
recog.I = uint8(zeros(Nx,Ny,M2)); % init for speed
for i = 1:M2
    [I,mp] = imread(fullfile(recog.path,di(i).name),imType);
    I = I(81:150,81:150,:);   %(66:185,76:175,:)
    recog.I(:,:,i) = rgb2gray(I);     % image
    %recog.I(:,:,i) = adapthisteq(I,'clipLimit',0.001);
    %recog.I(:,:,i) = imadjust(I);
    %recog.I(:,:,i) = histeq(I);
    recog.classNameTrue{i} = di(i).name(1:end-4); % specific for my naming
%     switch dbType
%         case 'ALAN'
             recog.classTrue{i} = ... % matches by filename
                 strmatch(recog.classNameTrue{i}(1:end-1),train.classNameTrue); 
%         case 'UMIST'
%             recog.classTrue{i} = ... % matches by filename
%                 strmatch(recog.classNameTrue{i}(1:2),train.classNameTrue); 
%     end
    recog.classEst(i) = NaN;
end

% plot input images
if plots.intermediateOn
    figure,montage(reshape(train.I,[Nx Ny 1 M]),mp),title('training images')
    if plots.savePlotsOn, saveas(gcf,'training_images','png'), end
    %could use >>unix('/sw/bin/montage -h')
	figure,montage(reshape(recog.I,[Nx Ny 1 M2]),mp),title('recog images')
    if plots.savePlotsOn, saveas(gcf,'recog_images','png'), end
end

% ------ Call Face Recog Alorithm ------
switch frAlgorithm
    case 'eigen'
        % -- Compute EigenFaces using "training" faces --
        % Outputs:
        %   P       => eigenfaces (eigenvectors)
        %   train.wt=> weights for each training face
        %   train.mean => average image
        [P,train] = computeEigenfaces(train,nEigFace,plots);
        
        % -- Classify the "recognition" faces --
		% Outputs:
		%   recog.classNameEst  => estimated class name
		%   recog.classEst      => estimated class number
		%   recog.isCorrectClass=> 1/0 for corr/incorr classification
        [recog] = classifyFaces(recog,train,P,threshFace,threshClass,plots);
        
    case 'fisher'
        % -- Compute EigenFaces (PCA) using "training" faces --
        %[P1,train1] = computeEigenfaces(train,nEigFace,plots);
        MpFish = M-length(trainClass.num);
        MpFish = min(MpFish,maxFishEigFace); % acb - impose face space limit
        [P1,train1] = computeEigenfaces(train,MpFish,plots);
        
        % -- Compute FisherFaces using "training" faces and "EigenFaces" --
        % Outputs:
        %   P2       => fisherfaces (eigenvectors)
        %   train2.wt => weights for each training face
        [P2,train2] = computeFisherfaces(train,trainClass,plots,P1);
        
        % -- Classify the "recognition" faces --
        [recog] = classifyFaces(recog,train2,P2,threshFace,threshClass,plots);
        
    otherwise
        error(['I don''t know face recognition algorithm "' ...
                frAlgorithm '"'])
end

% ------ Plot/Display Results ------

% Plot nice recog -> top matches figure with subplots!
if plots.finalOn
	%sc = [.5 .8]; % portion of screen [width height]  % [Left Bot Wid Hei]
	%figure('Units','Normalized','Position',[(1-sc)/2 sc],'Color',1*[1 1 1])
	nTopMatch = 3;
	nCh = 6;
	p = 1; % current subplot
    f = 0; % current subfigure
    nRowsMax = 7; % max input faces per subfigure
    nSubF = ceil(M2/nRowsMax);
	for i = 1:M2 % each recog image
        % create new subfigure when necessary
        if rem(i-1,nRowsMax) == 0
            f = f+1;
            hf(f) = subfigure(1,nSubF,f); set(hf,'Color',1*[1 1 1])
            p = 1;
        end
        
        % plot input image to be recognized
        subplot(nRowsMax,nTopMatch+2,p)
        imshow(recog.I(:,:,i),mp)
        if p==1, title('inputs'); end
        nc = min(nCh,length(recog.classNameTrue{i}));
        text(0,0,recog.classNameTrue{i}(1:nc),'FontSize',7,'Color','b')
        p = p+2;
        
        % plot closest matching images from training set
        [val,ndx] = sort(recog.euDis(:,i));
        for j = 1:nTopMatch
            subplot(nRowsMax,nTopMatch+2,p)
            imshow(train.I(:,:,ndx(j)),mp)
            if p==3 & j==1
                title(sprintf('top %d matches ...',nTopMatch));
            end
            if strncmp(train.classNameTrue{ndx(j)},recog.classNameTrue{i},2)
                c = 'b'; %[0 0 1];
            else
                c = 'r';
            end
            nc = min(nCh,length(train.classNameTrue{ndx(j)}));
            text(0,0,train.classNameTrue{ndx(j)}(1:nc),...
                'FontSize',7,'Color',c)
            p = p+1;
        end
	end
    
    if plots.savePlotsOn
        for i=1:f
            saveas(hf(i),sprintf('top3_matches_%d',i),'png')
        end
    end
end

%save workspace_dump
%disp('done')
