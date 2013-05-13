%%
%  Face Recognizer (Eigenfaces vs Fisherfaces)
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
height = 120;
width = 100;
image_dims = [height width];
r1 = (125 - height/2) + 1;
r2 = 125 + height/2;
c1 = (125 - width/2) + 1;
c2 = 125 + width/2;

% total training images have to be 150
train_images = zeros(prod(image_dims), N*3);
j = [1 51 101];
%disp_train_set = {};
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
    %disp_train_set{k} = disp_Timages;
    figure;
    montage(disp_Timages);
    %title(sprintf('Training SET %d- (Gray scale,%dx%d)',k,image_dims(1),image_dims(2)))
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
%title('Variance between Eigen faces')
hold off;

% We select 45 Eigen Vectors because that captures around 90%
% variance in the data
neigfaces = 45;
evec = eigvec(:,1:neigfaces);
%featurespace = evec'*mean_faces;

disp_eigvec = reshape(evec,[image_dims 1 neigfaces]);
for i = 1:neigfaces % scale for plot
    mx = max(evec(:,i));
    mi = min(evec(:,i));
    disp_eigvec(:,:,1,i) = (disp_eigvec(:,:,1,i)-mi)./(mx-mi);
end
figure;
montage(disp_eigvec); %title(sprintf('%d Eigen faces',neigfaces))

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
%featurespace2 = Eigvecs'*mean_faces;

disp_eigvec2 = reshape(Eigvecs,[image_dims 1 Classes-1]);
for i = 1:Classes-1 % scale for plot
    mx = max(Eigvecs(:,i));
    mi = min(Eigvecs(:,i));
    disp_eigvec2(:,:,1,i) = (disp_eigvec2(:,:,1,i)-mi)./(mx-mi);
end
figure;
montage(disp_eigvec2); %title(sprintf('%d Fisher faces',Classes-1))

norm_eigvals2 = Eigvals/sum(Eigvals);
figure;
plot(cumsum(norm_eigvals2),'LineWidth',2)
xlim([1 10]); ylim([0 1]); grid on;
xlabel('Number of Eigen vectors')
ylabel('Normalized Eigen values')
%title('Variance between Fisher faces')

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
probe_images = probe_images./256;

% Plot probe images
disp_Pimages = reshape(probe_images,[image_dims 1 N]);
for i = 1:N % scale for plot
    mx = max(probe_images(:,i));
    mi = min(probe_images(:,i));
    disp_Pimages(:,:,1,i) = (disp_Pimages(:,:,1,i)-mi)./(mx-mi);
end
figure;
montage(disp_Pimages);
%title(sprintf('Probe SET (Gray scale,%dx%d)',image_dims(1),image_dims(2)))

%% Classification Accuracy
%

m_faces_arr = train_images_arrange - repmat(mean_face,1,N*3);
featurespace = evec'*m_faces_arr;
featurespace2 = Eigvecs'*m_faces_arr;

mean_probe = probe_images - repmat(mean_face,1,N);
featurevec = evec'*mean_probe;
featurevec2 = Eigvecs'*mean_probe;

% 1-NN approach based on Euclidean distance
Euc_dist = zeros(N,N*3,2);

for i = 1:N  % Number of probe images
    for j = 1:N*3 % Number of training images
        Euc_dist(i,j,1) = sqrt(sum((featurespace(:,j) - featurevec(:,i)).^2));
        Euc_dist(i,j,2) = sqrt(sum((featurespace2(:,j) - featurevec2(:,i)).^2));
    end
end

False_pos = zeros(Classes,2);
ind1 = 1;
ind2 = 1;
% Thresholding values (Only needed when there are non-faces in probe set)
% thres1 = 20;    % Eigen faces
% thres2 = 5;     % Fisher faces
% for i = 1:5:N
% True_pos(ind2,1) = sum(sum(Euc_dist(i:i+4,ind1:ind1+14,1) < thres1));
% True_pos(ind2,2) = sum(sum(Euc_dist(i:i+4,ind1:ind1+14,2) < thres2));
% Idx = setdiff(1:150,ind1:ind1+14);
% False_pos(ind2,1) = sum(sum(Euc_dist(i:i+4,Idx,1) < thres1));
% False_pos(ind2,2) = sum(sum(Euc_dist(i:i+4,Idx,2) < thres2));
% ind1 = ind1 + 15;
% ind2 = ind2 + 1;
% end
% False_neg = 75 - True_pos;
% True_neg = 675 - False_pos;
%  True positive rate, False positive rate, Accuracy
% TPR = True_pos./75;
% FPR = False_pos./675;
% ACC = (True_pos + True_neg)./750;

for i = 1:5:N
    [minv1 mini1] = min(Euc_dist(i:i+4,:,1),[],2);
    [minv2 mini2] = min(Euc_dist(i:i+4,:,2),[],2);
    Idx = setdiff(1:150,ind1:ind1+14);
    False_pos(ind2,1) = sum(sum(repmat(mini1,1,135) == repmat(Idx,5,1),2));
    False_pos(ind2,2) = sum(sum(repmat(mini2,1,135) == repmat(Idx,5,1),2));
    ind1 = ind1 + 15;
    ind2 = ind2 + 1;
end
True_pos = 5 - False_pos;
False_neg = 5 - True_pos;
True_neg = True_pos;

% True positive rate, False positive rate, Accuracy
TPR = True_pos./5;
FPR = False_pos./5;
ACC = (True_pos + True_neg)./10;

%ACC1 = ACC;
%save accuracy ACC1

%% Plot ROC
%
figure;
mar = 'dv^><';
col = 'rgbmk';
hold on;
plot([0:1],[0:1],'r-','LineWidth',2)
H = zeros(1,10);
for i = 1:Classes
    ii = mod(i,5)+1;
    H(i) = plot(FPR(i,1),TPR(i,1),mar(ii),'Color',col(ii),'MarkerSize',6,...
    'LineWidth',4);
end
legend(H,'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','Location',...
    'EastOutside')
xlim([0 1]); ylim([0 1]); grid on;
xlabel('Flase positive rate')
ylabel('True positive rate')
title('ROC of Eigenface analysis')
hold off;

figure;
hold on;
plot([0:1],[0:1],'r-','LineWidth',2)
H = zeros(1,10);
for i = 1:Classes
    ii = mod(i,5)+1;
    H(i) = plot(FPR(i,2),TPR(i,2),mar(ii),'Color',col(ii),'MarkerSize',6,...
    'LineWidth',4);
end
legend(H,'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','Location',...
    'EastOutside')
xlim([0 1]); ylim([0 1]); grid on;
xlabel('Flase positive rate')
ylabel('True positive rate')
title('ROC of Fisherface analysis')
hold off;

%% Plot Accuracies
%
% this should be calculated with different permutations
% of training sets and test set
%
figure;
bar(ACC); colormap(summer);
legend('Eigenface','Fisherface')
xlabel('Classes')
ylabel('Accuracies')
title('Eigenface vs Fisherface')

% mean and standard deviation between different accuracies
mean_ACC_Eigen = mean(ACC(:,1))
mean_ACC_Fisher = mean(ACC(:,2))
std_Eigen = std(ACC(:,1))
std_Fisher = std(ACC(:,2))

%%
% Plot best matches
%
disp_probe = reshape(probe_images,[image_dims 50]);
disp_train = reshape(train_images_arrange,[image_dims 150]);
eind = zeros(N,1);
fisd = zeros(N,1);
for i = 1:N
    [val1 eig_ind] = min(Euc_dist(i,:,1));
    [val2 fish_ind] = min(Euc_dist(i,:,2));
    eind(i) = eig_ind;
    fisd(i) = fish_ind;
end

ii = 1;
for i = 1:5
    subplot(5,1,ii)
    imshow([disp_probe(:,:,i) disp_train(:,:,eind(i)) disp_train(:,:,fisd(i))])
    ii = ii + 1;
end

