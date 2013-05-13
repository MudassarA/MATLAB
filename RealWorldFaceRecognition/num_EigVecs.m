%%
%
% Performance with different number of Eigenfaces
%
%
N = 50;
Classes = 10;
train_images_arrange = zeros(size(train_images));
ind = 1;
ind2 = 1:15:150;
for i = 1:5:N
    t_imgs1 = train_images(:,i:i+4);
    t_imgs2 = train_images(:,i+50:i+50+4);
    t_imgs3 = train_images(:,i+100:i+100+4);
    t_imgs = [t_imgs1 t_imgs2 t_imgs3];
    train_images_arrange(:,ind2(ind):ind2(ind)+14) = t_imgs;
    ind = ind + 1;
end
m_faces_arr = train_images_arrange - repmat(mean_face,1,N*3);
mean_probe = probe_images - repmat(mean_face,1,N);

ACC = zeros(Classes,20);
ind = 1;
for neig = 5:5:100    % 50 to 100 percent variance in the data
    
    evec = eigvec(:,1:neig);
    featurespace = evec'*m_faces_arr;
    featurevec = evec'*mean_probe;
    % 1-NN approach based on Euclidean distance
    Euc_dist = zeros(N,N*3);
    
    for i = 1:N  % Number of probe images
        for j = 1:N*3 % Number of training images
            Euc_dist(i,j) = sqrt(sum((featurespace(:,j) - featurevec(:,i)).^2));
        end
    end
    
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
    TPR = True_pos./5;
    FPR = False_pos./5;
    ACC(:,ind) = (True_pos + True_neg)./10;
    ind = ind + 1;
end

mACC = mean(ACC)

%%
% Plot Eigenfaces vs accuracies
%

figure;
plot([5:5:100],mACC,'r-','LineWidth',2)
xlabel('Number of Eigen Vectors')
ylabel('Mean accuracy')
xlim([1 101]);

figure;
set(0,'DefaultAxesLineStyleOrder','-|--|:')
plot([5:5:100],ACC,'LineWidth',2)
xlabel('Number of Eigen Vectors')
ylabel('Accuracies of classes')
xlim([1 101]);
legend('class 1','class 2','class 3','class 4','class 5',...
    'class 6','class 7','class 8','class 9','class 10','Location',...
    'SouthEast')
