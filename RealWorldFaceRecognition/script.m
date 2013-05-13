%%
%  Face Recognizer script file
%
clear
close all
clc
% Directory paths
% Choose three paths for training sets
% and one path for probe set
images_path{1} = 'projectFaces/B/';
images_path{2} = 'projectFaces/C/';
images_path{3} = 'projectFaces/D/';
images_path{4} = 'projectFaces/A/';
ACC = zeros(10,2,4);
TPR = zeros(10,2,4);
FPR = zeros(10,2,4);

for ts = 1:length(images_path)

    %
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
    disp_train_set = {};
    INDEX = setdiff(1:4,ts)
    k2 = 0;
    for k = INDEX(1):INDEX(end)  % 3 training sets
        train_set = dir(fullfile(images_path{k}, ['*.' 'jpg']));
        if isempty(train_set)
            sprintf('warning: No image files found with %s\n',images_path{k})
        end
        % read images from directory
        ind = 0;
        k2 = mod(k2,3) + 1;
        for i = j(k2):N*k2
            ind = mod(ind,N) + 1;
            image_name1 = fullfile(images_path{k}, train_set(ind).name);
            img = imread(image_name1);
            img = rgb2gray(img);
            img = img(r1:r2,c1:c2);
            train_images(:,i) = img(:);
        end
        
        % Plot training images
%          disp_Timages = reshape(train_images(:,j(k):N*k),[image_dims 1 N]);
%          ind = 0;
%          for i = j(k):N*k % scale for plot
%              ind = mod(ind,N) + 1;
%              mx = max(train_images(:,i));
%              mi = min(train_images(:,i));
%              disp_Timages(:,:,1,ind) = (disp_Timages(:,:,1,ind)-mi)./(mx-mi);
%          end
%          disp_train_set{k} = disp_Timages;
%          figure;
%          montage(disp_Timages);
%          title(sprintf('Training SET %d- (Gray scale,%dx%d)',k,image_dims(1),image_dims(2)))
    end

    % Eigen Faces
    %
    % Principal components analysis
    %
    train_images = train_images./256;  % Normalizing with gray scale
    mean_face = mean(train_images,2);
    mean_faces = train_images - repmat(mean_face,1,N*3); % normalized faces

    %[eigvec score eigval] = princomp(mean_faces');
    [eigvec eigval V] = svd(mean_faces,0);
    clear V;

%     norm_eigvals = eigval/sum(eigval);
%     figure;
%     plot(cumsum(norm_eigvals),'LineWidth',2)
%     hold on;
%     plot([45 45],[0.001 0.999],'r-','LineWidth',2)
%     xlim([1 150]); ylim([0 1]); grid on;
%     xlabel('Number of Eigen vectors')
%     ylabel('Normalized Eigen values')
%     title('Variance between Eigen faces')
%     hold off;

    % We select 45 Eigen Vectors because that captures around 90%
    % variance in the data
    neigfaces = 45;
    evec = eigvec(:,1:neigfaces);
    
    %featurespace = evec'*mean_faces;

%     disp_eigvec = reshape(evec,[image_dims 1 neigfaces]);
%     for i = 1:neigfaces % scale for plot
%         mx = max(evec(:,i));
%         mi = min(evec(:,i));
%         disp_eigvec(:,:,1,i) = (disp_eigvec(:,:,1,i)-mi)./(mx-mi);
%     end
%     figure;
%     montage(disp_eigvec); title(sprintf('%d Eigen faces',neigfaces))

    % Fisher Faces
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
    indd = 1;
    indd2 = 1:15:150;
    fprintf('computing the between class scatter...\n')
    for i2 = 1:5:Classes*5
        t_imgs1 = train_images(:,i2:i2+4);
        t_imgs2 = train_images(:,i2+50:i2+50+4);
        t_imgs3 = train_images(:,i2+100:i2+100+4);
        t_imgs = [t_imgs1 t_imgs2 t_imgs3];
        mean_classes(:,indd) = mean(t_imgs,2);
        train_images_arrange(:,indd2(indd):indd2(indd)+14) = t_imgs;
        indd = indd + 1;
    end
    for i3 = 1:Classes
        tmp1 = mean_classes(:,i3) - mean_face;
        tmp2 = tmp1 * tmp1';
        bc_scatter = bc_scatter + tmp2;
        fprintf('Class %d is done\n',i3)
    end

    % Within class scatter
    fprintf('computing the with in class scatter...\n')
    wc_scatter = zeros(prod(image_dims));
    indd = 1;
    for i4 = 1:15:Classes*5*3
        for j = i4:i4+14
            tmp1 = train_images_arrange(:,j) - mean_classes(:,indd);
            tmp2 = tmp1 * tmp1';
            wc_scatter = wc_scatter + tmp2;
        end
        fprintf('Class %d is done\n',indd)
        indd = indd + 1;
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

%     disp_eigvec2 = reshape(Eigvecs,[image_dims 1 Classes-1]);
%     for i = 1:Classes-1 % scale for plot
%         mx = max(Eigvecs(:,i));
%         mi = min(Eigvecs(:,i));
%         disp_eigvec2(:,:,1,i) = (disp_eigvec2(:,:,1,i)-mi)./(mx-mi);
%     end
%     figure;
%     montage(disp_eigvec2); title(sprintf('%d Fisher faces',Classes-1))
% 
%     norm_eigvals2 = Eigvals/sum(Eigvals);
%     figure;
%     plot(cumsum(norm_eigvals2))
%     xlim([1 10]); ylim([0 1]); grid on;
%     xlabel('Number of Eigen vectors')
%     ylabel('Normalized Eigen values')
%     title('Variance between Fisher faces')

    %
    % Probe set images
    % Read from directory
    %
    probe_set = dir(fullfile(images_path{ts}, ['*.' 'jpg']));

    if isempty(probe_set)
        sprintf('warning: No image files found with %s\n',images_path{ts})
    end

    % Input probe images from directory
    M = length(probe_set);
    probe_images = zeros(prod(image_dims), M);
    for i = 1:M
        image_name2 = fullfile(images_path{ts}, probe_set(i).name);
        img = imread(image_name2);
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
     title(sprintf('Probe SET (Gray scale,%dx%d)',image_dims(1),image_dims(2)))

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

    %True_pos = zeros(Classes,2);
    False_pos = zeros(Classes,2);
    ind1 = 1;
    ind2 = 1;
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
    TPR(:,:,ts) = True_pos./5;
    FPR(:,:,ts) = False_pos./5;
    ACC(:,:,ts) = (True_pos + True_neg)./10;
    
    clear Sbc Swc
    pause;
end

%% Plot ROC
%
figure;
mar = 'dv^><';
col = 'rgbk';
hold on;
plot([0:1],[0:1],'r-','LineWidth',2)
H = zeros(1,10);
for i = 1:Classes
    ii = mod(i,5)+1;
    jj = mod(i,4)+1;
    H(i) = plot(FPR(i,1,4),TPR(i,1,4),mar(ii),'Color',col(jj),'MarkerSize',6,...
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
    jj = mod(i,4)+1;
    H(i) = plot(FPR(i,2,4),TPR(i,2,4),mar(ii),'Color',col(jj),'MarkerSize',6,...
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
%subplot(221)
bar([ACCS(:,:,1) ACClbp(:,1)]); colormap(jet);
legend('Eigenface','Fisherface','LBP')
xlabel('Classes')
ylabel('Accuracies')


%% mean and standard deviation between different accuracies
mean_ACC_Eigen = zeros(4,1);
mean_ACC_Fisher = zeros(4,1);
mean_ACC_LBP = zeros(4,1);
std_Eigen = zeros(4,1);
std_Fisher = zeros(4,1);
std_LBP = zeros(4,1);
for i = 1:4
    mean_ACC_Eigen(i) = mean(ACCS(:,1,i));
    mean_ACC_Fisher(i) = mean(ACCS(:,2,i));
    mean_ACC_LBP(i) = mean(ACClbp(:,i));
    std_Eigen(i) = std(ACCS(:,1,i));
    std_Fisher(i) = std(ACCS(:,2,i));
    std_LBP(i) = std(ACClbp(:,i));
end
figure;
%subplot(121)
bar([mean_ACC_Eigen mean_ACC_Fisher mean_ACC_LBP]);
ylim([0 1]); set(gca,'XTickLabel',['A';'B';'C';'D']);
legend('Eigenface','Fisherface','LBP')
xlabel('Probe set')
ylabel('Mean accuracy')
%title('Mean accuracies over all classes')
figure;
%subplot(122)
bar([std_Eigen std_Fisher std_LBP]);
ylim([0 0.5]); set(gca,'XTickLabel',['A';'B';'C';'D']);
legend('Eigenface','Fisherface','LBP')
xlabel('Probe set')
ylabel('Standard deviation in accuracy')
%title('Standard deviation over all classes')
