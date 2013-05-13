%%
% Compute EigenFaces
%

function [P,train] = computeEigenfaces(train,Mp,plots)
% Outputs:
%   P       => eigenfaces
%   train.wt=> weights for each training face
%   train.mean => average image

% Find the image size, [Nx Ny], and the number of training images, M
[Nx Ny M] = size(train.I);

if Mp>M
    warning(sprintf(...
        ['Can''t use more principal comp than input imgaes!\n'...
            '  -> Using %d components.'],M))
    Mp = M;
end

% Use cached version of slow eigenvalue computations if found
svdFile = fullfile(train.path,'storedSvd.mat');
foundCache = exist(svdFile,'file');
if foundCache & ~train.forceNewSvd 
    % loads stored vars
    load(svdFile)
else
	% Compute EigenFaces using "training" faces
	% learn principal components from {x1, x2, ..., xn}
	% (1) find mean face, me, and
	%     differences from means, A
	X = double(reshape(train.I,[Nx*Ny M]))./256; % 1 column per face
	me = mean(X,2);
	A = X - repmat(me,[1 M]);
    clear X
	
	% (2) covariance matrix, S = A*A' (skip computing by using svd)
	% (3) partial eigenvalue decomposition S = U'*E*U
    [U,E,V] = svd(A,0); % singular val decomp much faster
	
	% (4) get sorted eigenvalues (diag of E) and eigenvectors (U)
	eigVals = diag(E);
	eigVecs = U;
    clear U V
    
    % store cache for future runs
    save(svdFile,'eigVecs','eigVals','me','A')
end
	
% (5) P' = [u1' u2' ... um'] % pick Mp principal components
P = eigVecs(:,1:Mp);        % ouput eigenfaces
lambda = eigVals(1:Mp);     % output weights

train.mean = me;

% Project each face in training set onto eigenfaces, storing weight
train.wt = P'*A;

% Reconstruct projected faces
R = P*train.wt + repmat(train.mean,[1 M]);

% Plot average face, eigenvals
if plots.intermediateOn % >> help truesize
	figure,imshow(reshape(train.mean,[Nx Ny])),title('avg face')
    if plots.savePlotsOn, saveas(gcf,'avg_face','png'), end
    figure,plot([1:length(eigVals)], eigVals,'x-'),title('\lambda strength')
    if plots.savePlotsOn, saveas(gcf,'eigval_strength','png'), end
end

% Plot eigenfaces
if plots.intermediateOn
	I = reshape(P,[Nx Ny 1 Mp]);
	for i = 1:Mp % scale for plot
        mx = max(P(:,i));
        mi = min(P(:,i));
        I(:,:,1,i) = (I(:,:,1,i)-mi)./(mx-mi);
	end 
	figure,montage(I),title('eigenfaces'); % eigenfaces
    if plots.savePlotsOn, saveas(gcf,'eigenfaces','png'), end
end
err = sum(eigVals(Mp+1:M).^2);

% Plot reconstructed images
if plots.intermediateOn
	I = reshape(R,[Nx Ny 1 M]);
	for i = 1:M % scale for plot
        mx = max(R(:,i));
        mi = min(R(:,i));
        I(:,:,1,i) = (I(:,:,1,i)-mi)./(mx-mi);
	end 
	figure,montage(I),title('reconst training images')
    if plots.savePlotsOn, saveas(gcf,'reconst_training_images','png'), end
end
