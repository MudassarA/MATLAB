%%
%  Face Recognizer (Eigen faces)
%
clear
close all
clc
% Directory paths
% Choose first three paths for training sets
images_path{1} = 'projectFaces/A/';
images_path{2} = 'projectFaces/B/';
images_path{3} = 'projectFaces/C/';
% one path for probe set
images_path{4} = 'projectFaces/D/';

%%
% Read images from directory
%
N = 50;        % single training set sample images
% select image dimensions
height = 100;
width = 100;
image_dims = [height width];
r1 = (125 - height/2) + 1;
r2 = 125 + height/2;
c1 = (125 - width/2) + 1;
c2 = 125 + width/2;

% total training images have to be 150
train_images = zeros(prod(image_dims), N*3);
j = [1 51 101];
disp_train_set = {};
for k = 1:3  % 3 training sets
    train_set = dir(fullfile(images_path{k}, ['*.' 'jpg']));
    if isempty(train_set)
        sprintf('warning: No image files found with %s\n',images_path{k})
    end
    % read images from directory
    ind = 0;
    for i = j(k):N*k
        ind = mod(ind,N) + 1;
        image_name = fullfile(images_path{k}, train_set(ind).name);
        img = imread(image_name);
        img = rgb2gray(img);
        img = img(r1:r2,c1:c2);
        train_images(:,i) = img(:);
    end
    
    % Plot training images
    disp_Timages = reshape(train_images(:,j(k):N*k),[image_dims 1 N]);
    ind = 0;
    for i = j(k):N*k % scale for plot
        ind = mod(ind,N) + 1;
        mx = max(train_images(:,i));
        mi = min(train_images(:,i));
        disp_Timages(:,:,1,ind) = (disp_Timages(:,:,1,ind)-mi)./(mx-mi);
    end
    disp_train_set{k} = disp_Timages;
    figure;
    montage(disp_Timages);
    title(sprintf('Training SET %d- (Gray scale,%dx%d)',k,image_dims(1),image_dims(2)))
end

%% Eigen Faces
%
% Principal components analysis
%
train_images = train_images./256;  % Normalizing with gray scale
mean_face = mean(train_images,2);
mean_faces = train_images - repmat(mean_face,1,N*3); % normalized faces

%[eigvec score eigval] = princomp(mean_faces');
[eigvec eigval V] = svd(mean_faces,0);
clear V;

norm_eigvals = eigval/sum(eigval);
figure;
plot(cumsum(norm_eigvals),'LineWidth',2)
hold on;
plot([45 45],[0.001 0.999],'r-','LineWidth',2)
xlim([1 150]); ylim([0 1]); grid on;
xlabel('Number of Eigen vectors')
ylabel('Normalized Eigen values')
title('Variance between Eigen faces')

% We select 45 Eigen Vectors because that captures around 90%
% variance in the data
neigfaces = 45;
evec = eigvec(:,1:neigfaces);
featurespace = evec'*mean_faces;

disp_eigvec = reshape(evec,[image_dims 1 neigfaces]);
for i = 1:neigfaces % scale for plot
    mx = max(evec(:,i));
    mi = min(evec(:,i));
    disp_eigvec(:,:,1,i) = (disp_eigvec(:,:,1,i)-mi)./(mx-mi);
end
figure;
montage(disp_eigvec); title(sprintf('%d Eigen faces',neigfaces))

%% Fisher Faces
%
% Linear Discriminant Analysis
%
tic
fprintf('Computing Finsherfaces... \n')
Classes = 10;
mean_classes = zeros(prod(image_dims),Classes);
% Between class scatter
bc_scatter = zeros(prod(image_dims));
train_images_arrange = zeros(size(train_images));
ind = 1;
ind2 = 1:15:150;
fprintf('computing the between class scatter...\n')
for i = 1:5:Classes*5
    t_imgs1 = train_images(:,i:i+4);
    t_imgs2 = train_images(:,i+50:i+50+4);
    t_imgs3 = train_images(:,i+100:i+100+4);
    t_imgs = [t_imgs1 t_imgs2 t_imgs3];
    mean_classes(:,ind) = mean(t_imgs,2);
    train_images_arrange(:,ind2(ind):ind2(ind)+14) = t_imgs;
    ind = ind + 1;
end
for i = 1:Classes
    tmp1 = mean_classes(:,i) - mean_face;
    tmp2 = tmp1 * tmp1';
    bc_scatter = bc_scatter + tmp2;
    fprintf('Class %d is done\n',i)
end

% Within class scatter
fprintf('computing the with in class scatter...\n')
wc_scatter = zeros(prod(image_dims));
ind = 1;
for i = 1:15:Classes*5*3
    for j = i:i+14
        tmp1 = train_images_arrange(:,j) - mean_classes(:,ind);
        tmp2 = tmp1 * tmp1';
        wc_scatter = wc_scatter + tmp2;
    end
    fprintf('Class %d is done\n',ind)
    ind = ind + 1;
end
clear tmp1 tmp2

% In order to obtain Fisherfaces, inverse of with in class
% scatter is required, i.e., wc_scatter^-1. If the sample
% feature vectors are defined in a p-dimensional space and
% p is larger than the total number of samples n , then
% wc_scatter is singular therefore we project the sample
% vectors into PCA subspace of r-dimensions, r <= rank(wc_scatter)

% Compute eig values and eig vectors
Sbc = evec.'*bc_scatter*evec;
Swc = evec.'*wc_scatter*evec;
clear bc_scatter wc_scatter
fprintf('Computing PCA+LDA \n')
[V D] = eig(Sbc,Swc);             % PCA + LDA
Ds = diag(D);
[sval sind] = sort(abs(Ds));
sind = flipud(sind);
Eigvals = Ds(sind);
Eigvecs = evec*V(:,sind);
clear V D Ds

t = toc;
fprintf('Elapsed Time\n')
disp(datestr(datenum(0,0,0,0,0,t),'HH:MM:SS'))

% Only Classes - 1 weights
Eigvals = Eigvals(1:Classes-1);
Eigvecs = Eigvecs(:,1:Classes-1);
Eigvecs = Eigvecs./repmat(sqrt(sum(Eigvecs.^2)),prod(image_dims),1);
featurespace2 = Eigvecs'*mean_faces;

disp_eigvec2 = reshape(Eigvecs,[image_dims 1 Classes-1]);
for i = 1:Classes-1 % scale for plot
    mx = max(Eigvecs(:,i));
    mi = min(Eigvecs(:,i));
    disp_eigvec2(:,:,1,i) = (disp_eigvec2(:,:,1,i)-mi)./(mx-mi);
end
figure;
montage(disp_eigvec2); title(sprintf('%d Fisher faces',Classes-1))

norm_eigvals2 = Eigvals/sum(Eigvals);
figure;
plot(cumsum(norm_eigvals2))
xlim([1 10]); ylim([0 1]); grid on;
xlabel('Number of Eigen vectors')
ylabel('Normalized Eigen values')
title('Variance between Fisher faces')

%%
% Probe set images
% Read from directory
%probe_imgs_path = 'projectFaces/D/';
probe_set = dir(fullfile(images_path{4}, ['*.' 'jpg']));

if isempty(probe_set)
    sprintf('warning: No image files found with %s\n',images_path{4})
end

% Input probe images from directory
M = length(probe_set);
probe_images = zeros(prod(image_dims), M);
for i = 1:M
    image_name = fullfile(images_path{4}, probe_set(i).name);
    img = imread(image_name);
    img = rgb2gray(img);
    img = img(r1:r2,c1:c2);
    probe_images(:,i) = img(:);
end

% Plot probe images
disp_Pimages = reshape(probe_images,[image_dims 1 N]);
for i = 1:N % scale for plot
    mx = max(probe_images(:,i));
    mi = min(probe_images(:,i));
    disp_Pimages(:,:,1,i) = (disp_Pimages(:,:,1,i)-mi)./(mx-mi);
end
figure;
montage(disp_Pimages);
title(sprintf('Probe SET (Gray scale,%dx%d)',image_dims(1),image_dims(2)))

%% Classification
%
% think about featurevecs and featurespace

mean_probe = probe_images(:,18) - mean_face;
featurevec = evec'*(mean_probe);
featurevec2 = Eigvecs'*(mean_probe);
sim_score = zeros(2,N*3);
for i = 1:N*3
    sim_score(1,i) = 1/(1 + norm(featurespace(:,i) - featurevec));
    sim_score(2,i) = 1/(1 + norm(featurespace2(:,i) - featurevec2));
end
[mscore1 mind1] = max(sim_score(1,:))
[mscore2 mind2] = max(sim_score(2,:))
set1 = 1;
if mind1 > 50
    mind1 = mind1 - 50;
    set1 = 2;
elseif mind1 > 100
    mind1 = mind1 - 100;
    set1 = 3;
end
set2 = 1;
if mind2 > 50
    mind2 = mind2 - 50;
    set2 = 2;
elseif mind2 > 100
    mind2 = mind2 - 100;
    set2 = 3;
end

figure; imshow([disp_Pimages(:,:,1,18) disp_train_set{set1}(:,:,1,mind1)]),...
    title('EigenFace Result for single probe image')
figure; imshow([disp_Pimages(:,:,1,18) disp_train_set{set2}(:,:,1,mind2)]),...
    title('FisherFace Result for single probe image')
