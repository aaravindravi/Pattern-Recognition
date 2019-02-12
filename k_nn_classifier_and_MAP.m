%% K Nearest Neighbour Classifier

clear all
close all

randn('seed',0);
N = 200;

muA = [0 0]';
muB = [3 0]';

covA = [3 1; 1 2];
covB = [7 -3;-3 4];


xA = mvnrnd(muA,covA,N);
xB = mvnrnd(muB,covB,N);

m(:,:,1)=muA;
m(:,:,2)=muB;

S(:,:,1) = covA;
S(:,:,2) = covB;

%Class probability
pw = [0.5 0.5];

figure('units','normalized','outerposition',[0 0 1 1])

figure(1);
subplot(1,3,1);
plot(xA(:,1),xA(:,2),'.r');
hold on;
plot(xB(:,1),xB(:,2),'.b')
axis equal

%% Create sample test points to classify in the right class
boundary = [0 0];
MAPboundary = [0 0];
[X1 X2] = meshgrid(-10:0.5:10, -10:0.5:10);

%% Classify sample Test Points in one of the classes 
   K = 1;
    for vectors = 1:length(X1)

        sampleData = [X1(vectors,:);X2(vectors,:)];

        %Nearest Neighbor Classifier
        [IDXA,DA] = knnsearch(xA,sampleData','K',K);
        [IDXB,DB] = knnsearch(xB,sampleData','K',K);
        differencesDist = DA-DB;
        boundaryIndices = find(differencesDist == 0);
        boundary = [boundary;[sampleData(:,boundaryIndices)]'];
        
        %MAP Classification
        pc = bayes_class(m,S,pw,sampleData); %can specify the number of classes or read from the variables
        differencesProb = pc(:,1)-pc(:,2);
        MAPboundaryIndices = find(differencesProb == 0);
        MAPboundary = [MAPboundary;[sampleData(:,MAPboundaryIndices)]'];


        %% Plotting the newly classified points
        subplot(1,3,1);
        hold on
        class1Indices = find(sum(sign(differencesDist),2)>0);
        plot(sampleData(1,class1Indices),sampleData(2,class1Indices),'.g');
        class2Indices = find(sum(sign(differencesDist),2)<0);
        plot(sampleData(1,class2Indices),sampleData(2,class2Indices),'.m');
        title(strcat(num2str(K),'-Nearest Neighbour Classifier'));
        axis equal
        axis([-10 10 -10 10]);
        hold off;
        %% Plotting the MAP classified points

        subplot(1,3,2);
        hold on;
        class1MAPIndices = find(differencesProb < 0);
        plot(sampleData(1,class1MAPIndices),sampleData(2,class1MAPIndices),'.g');
        class2MAPIndices = find(differencesProb > 0);
        plot(sampleData(1,class2MAPIndices),sampleData(2,class2MAPIndices),'.m');
        plot(MAPboundary(2:end,1),MAPboundary(2:end,2),'.k');
        plot(xA(:,1),xA(:,2),'.r');
        plot(xB(:,1),xB(:,2),'.b');
        title('MAP Classifier');
        axis equal
        axis([-10 10 -10 10]);
        hold off;

    end
       
 
%% Plotting the Contours

subplot(1,3,3);
plot(boundary(2:end,1),boundary(2:end,2),'k');
hold on

X3A = mvnpdf([X1(:),X2(:)],muA',covA);
X3B = mvnpdf([X1(:),X2(:)],muB',covB);
X3A = reshape(X3A,length(X2),length(X1));
X3B = reshape(X3B,length(X2),length(X1));

%surf(X1,X2,X3A);
contour(X1,X2,X3A);
contour(X1,X2,X3B);
axis equal
title ('Contour Plot')
axis ([-10 10 -10 10]);
hold off
saveas(gcf,strcat(num2str(K),'_Nearest_Neighbour_Classifier_with_MAP.png'));

      
        
