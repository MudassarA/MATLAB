%%
% Local binary pattern feature extraction
%
clear
%clc
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
tic
mapping = getmapping(24,'riu2');  % uniform rotation-invariant LBP (riu2)
                               % rotation-invariant (ri)
                               % uniform LBP (u2)
t = toc;
fprintf('Elapsed Time\n')
disp(datestr(datenum(0,0,0,0,0,t),'HH:MM:SS'))

%%
HT = zeros(26,N*3);   % set row size depending on LBP output... 10,36,59,18,26
ind = 1;
%HT = [];
for i = 1:3 % number of training sets
    train_set = dir(fullfile(images_path{i}, ['*.' 'jpg']));
    if isempty(train_set)
        sprintf('warning: No image files found with %s\n',images_path{i})
    end
    for j = 1:N
        image_name = fullfile(images_path{i}, train_set(j).name);
        img = imread(image_name);
        img = rgb2gray(img);
        img = img(r1:r2,c1:c2);
        HT(:,ind) = lbp(img,2,24,mapping,'nh');    % normalized histogram features
        ind = ind + 1;
        %h = lbp(img);
        %HT = [HT h'];
    end
end
% Compute LBP features for probe set
m = size(HT,1);
HP = zeros(m,N);
train_set = dir(fullfile(images_path{4}, ['*.' 'jpg']));
if isempty(train_set)
    sprintf('warning: No image files found with %s\n',images_path{4})
end
for k = 1:N
    image_name = fullfile(images_path{4}, train_set(k).name);
    img = imread(image_name);
    img = rgb2gray(img);
    img = img(r1:r2,c1:c2);
    HP(:,k) = lbp(img,2,24,mapping,'nh');    % normalized histogram features
    %HP(:,k) = lbp(img);
end

%%
% 1-NN classification
%
Classes = 10;

% arrange training features in order of classes
arrange_HT = zeros(size(HT));
ind1 = 1;
ind2 = 1:15:150;
for i = 1:5:Classes*5
    ht_1 = HT(:,i:i+4);
    ht_2 = HT(:,i+50:i+50+4);
    ht_3 = HT(:,i+100:i+100+4);
    t_imgs = [ht_1 ht_2 ht_3];
    arrange_HT(:,ind2(ind1):ind2(ind1)+14) = t_imgs;
    ind1 = ind1 + 1;
end

% compute the Euclidean distance
Euc_dist = zeros(N,N*3);
for i = 1:N  % Number of probe images
    for j = 1:N*3 % Number of training images
        Euc_dist(i,j) = sqrt(sum((arrange_HT(:,j) - HP(:,i)).^2));
    end
end

%
% Accuracies
%
False_pos = zeros(Classes,1);
ind1 = 1;
ind2 = 1;
for i = 1:5:N
    [minv mini] = min(Euc_dist(i:i+4,:),[],2);
    Idx = setdiff(1:150,ind1:ind1+14);
    False_pos(ind2,1) = sum(sum(repmat(mini,1,135) == repmat(Idx,5,1),2));
    ind1 = ind1 + 15;
    ind2 = ind2 + 1;
end
True_pos = 5 - False_pos;
False_neg = 5 - True_pos;
True_neg = True_pos;

% True positive rate, False positive rate, Accuracy
TPR = True_pos/5;
FPR = False_pos/5;
ACC = (True_pos + True_neg)/10

%%
%
Classes = 10;
figure;
mar = 'dv^><';
col = 'rgbk';
hold on;
plot([0:1],[0:1],'r-','LineWidth',2)
H = zeros(1,10);
for i = 1:Classes
    ii = mod(i,5)+1;
    jj = mod(i,4)+1;
    H(i) = plot(FPR(i,1),TPR(i,1),mar(ii),'Color',col(jj),'MarkerSize',6,...
    'LineWidth',4);
end
legend(H,'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','Location',...
    'EastOutside')
xlim([0 1]); ylim([0 1]); grid on;
xlabel('Flase positive rate')
ylabel('True positive rate')
title('ROC of LBP')
hold off;
