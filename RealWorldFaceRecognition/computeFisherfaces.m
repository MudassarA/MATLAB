%
% Compute Fisher Faces
%

function [P,train] = computeFisherfaces(train,trainClass,plots,P1)
% Outputs:
%   P       => fisherfaces
%   train.wt=> weights for each training face
%   train.mean => average image
% Summary:
%   The idea behind this approach is the maximize the ratio of
% between-class scatter to that of within-class scatter.

% Find the image size, [Nx Ny], and the number of training images, M
[Nx Ny M] = size(train.I);

% Find mean face, me, and
%     differences from means, A
X = double(reshape(train.I,[Nx*Ny M]))./256; % 1 column per face
me = mean(X,2);
A = X - repmat(me,[1 M]);

% Use cached version of slow scattermatrix computations if found
scatFile = fullfile(train.path,'storedScat.mat');
foundCache = exist(scatFile,'file');
numOfClass = length(trainClass.num);
if foundCache & ~train.forceNewScat 
    % loads stored vars
    load(scatFile)
else
	% Calculate the between-class scatter matrix, Sb
	%       and the within-class scatter matrix,  Sw
	prod = zeros(Nx*Ny);
	Sb = zeros(Nx*Ny);
	for i = 1:numOfClass
        row = trainClass.mean{i} - me;
        prod = row * row';
        Sb = Sb + prod;
	end
	
	Sw = zeros(Nx*Ny);
	for i = 1:numOfClass
        for j = (trainClass.classStartIndex(i)):(trainClass.classEndIndex(i))
            row = X(:,j) - trainClass.mean{i};
            prod = row * row';
            Sw = Sw + prod;
        end
	end
    clear prod row
    
    % store cache for future runs
    save(scatFile,'Sb','Sw')
end

% Use PCA to project into subspace
Sbb = P1.'*Sb*P1; 
Sww = P1.'*Sw*P1;
clear Sb Sw % save memory

% Current decomposition method: (from class)
% Find generalized eigenvalues & eigenvectors using eig(A,B)
[V,D] = eig(Sbb,Sww);

% Another possible method: (from class)
% 1. Note that we only care about the direction of Sw*W on m1-m2
% 2. Guess w = Sw^-1 * (m1-m2), then iterate ???

% One more possible method: (from Duda book)
% 1. Find the eigenvalues as the roots of the characteristic
%    polynomial:  
%       det(Sb - lambda(i)*Sw) = 0
% 2. Then solve for the eigenvectors w(i) directly using:
%       (Sb - lambda(i)*Sw)*w(i) = 0

% Extract eigenvalues and sort largest to smallest
Ds = diag(D)
[tmp,ndx] = sort(abs(Ds));
ndx = flipud(ndx);
% get sorted eigenvalues (diag of E) and 
% eigenvectors (project V back into full space using P1)
eigVals = Ds(ndx)
eigVecs = P1*V(:,ndx);
clear D Ds V % save a little memory

% Only keep numOfClass-1 weights

% Only keep numOfClass-1 weights, and
% Scale to make eigenvectors normalized => sum(P(:,1).^2)==1
Mp = numOfClass-1;      
lambda = eigVals(1:Mp); % output weights
P = eigVecs(:,1:Mp);    % ouput fisherfaces
P = P./repmat(sum(P.^2).^0.5,Nx*Ny,1); % normalize

train.mean = me;

% Project each face in training set onto fisherfaces, storing weight
train.wt = P.'*A;

% Reconstruct projected faces
R = P*train.wt + repmat(train.mean,[1 M]);

% Plot average face, eigenvals
if plots.intermediateOn % >> help truesize
    figure,plot([1:length(eigVals)], eigVals,'x-'),title('\lambda strength')
    if plots.savePlotsOn, saveas(gcf,'fish_eigval_strength','png'), end
end

% Plot fisherfaces
if plots.intermediateOn
	I = reshape(P,[Nx Ny 1 Mp]);
	for i = 1:Mp % scale for plot
        mx = max(P(:,i));
        mi = min(P(:,i));
        I(:,:,1,i) = (I(:,:,1,i)-mi)./(mx-mi);
	end 
	figure,montage(I),title('fisherfaces') % fisherfaces
    if plots.savePlotsOn, saveas(gcf,'fisherfaces','png'), end
end

% Plot reconstructed images
if plots.intermediateOn
	I = reshape(R,[Nx Ny 1 M]);
	for i = 1:M % scale for plot
        mx = max(R(:,i));
        mi = min(R(:,i));
        I(:,:,1,i) = (I(:,:,1,i)-mi)./(mx-mi);
	end 
	figure,montage(I),title('reconst training images')
    if plots.savePlotsOn, saveas(gcf,'fish_reconst_trainign_images','png')
    end
    
	%figure
	%imagesc(I(:,:,1,1))
    %set(gca,'Units','pixels','Position',[100 100 3*[Ny Nx]])
	%colormap gray
end
