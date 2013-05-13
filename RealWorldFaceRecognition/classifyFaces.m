%
%  Classify Faces
%

function [recog] = classifyFaces(recog,train,P,threshFace,threshClass,plots)
% Outputs:
%   recog.classNameEst  => estimated class name
%   recog.classEst      => estimated class number
%   recog.isCorrectClass=> 1 or 0 for correct/incorrect classification

% Find the image size, [Nx Ny], and the number of training images, M
[Nx Ny M] = size(train.I);
% Find the image size, [Nx Ny], and the number of recog images, M2
[Nx Ny M2] = size(recog.I);

% Init some values
Mp = length(P(1,:));
X2 = double(reshape(recog.I,[Nx*Ny M2]))./256; % 1 column per face
A2 = X2 - repmat(train.mean,[1 M2]);
% Project each face in recog set onto eigenfaces, storing weight
recog.wt = P'*A2;
% Reconstruct projected faces
R = P*recog.wt + repmat(train.mean,[1 M2]);

% Plot reconstructed images
if plots.intermediateOn
	I = reshape(R,[Nx Ny 1 M2]);
	for i = 1:M2 % scale for plot
        mx = max(R(:,i));
        mi = min(R(:,i));
        I(:,:,1,i) = (I(:,:,1,i)-mi)./(mx-mi);
	end 
	figure,montage(I),title('reconst recog images')
    if plots.savePlotsOn, saveas(gcf,'reconst_recog_images','png'), end
end

% Find euclidian distance from each recog face to each known face
recog.euDis = zeros(M,M2);
for i = 1:M2 % each recog face
    for j = 1:M % each known face class
        recog.euDis(j,i) = sqrt(sum((recog.wt(:,i) - train.wt(:,j)).^2));
    end
end

% Classifiy to Nearest-Neighbor with two thresholds:
%   threshFace => how close a euDis has to be to any face?
%   threshWho  => how close a euDis has to be to a training face to
%                 declare match
[minDis ndx] = min(recog.euDis);
	%recog.classNameTrue % truth
	%train.classNameTrue(ndx) % estimated classification
fprintf('Results of face recognition:\n')
for i = 1:M2 % each recog face
    if minDis(i) > threshFace
        recog.classNameEst{i} = 'NonFace';
    elseif minDis(i) > threshClass
        recog.classNameEst{i} = 'UnknownFace';
    else
        recog.classNameEst{i} = train.classNameTrue{ndx(i)};
        recog.classEst(i) = ndx(i);
    end
    
    recog.isCorrectClass(i) = ...
        any(recog.classEst(i) == recog.classTrue{i});
    fprintf('\trecognized %s as %s\n',...
        recog.classNameTrue{i},recog.classNameEst{i});
end

numCorrect = sum(recog.isCorrectClass);
fprintf('\t%d of %d (%d%%) faces correctly classified\n',...
    numCorrect,M2,round(numCorrect/M2*100));
