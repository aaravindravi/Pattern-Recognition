%% Pattern Recognition Exercises

clear all
close all

randn('seed',0);
N = 200;


% Case 1
testCase = 1;
muA = [0 0]';
muB = [3 0]';

covA = eye(2);
covB = eye(2);



% Case 2
testCase = 2;

muA = [-1 0]';
muB = [1 0]';

covA = [4 3; 3 4];
covB = [4 3; 3 4];


% Case 3
testCase = 3;

muA = [0 0]';
muB = [3 0]';

covA = [3 1; 1 2];
covB = [7 -3;-3 4];



%% Case 4

testCase =4;

load('data.mat');

[muA,covA] = gaussian_ml_estimator(a);
[muB,covB] = gaussian_ml_estimator(b);


m(:,:,1)=muA;
m(:,:,2)=muB;

S(:,:,1) = covA;
S(:,:,2) = covB;

%Class probability
pw = [0.5 0.5];

if (testCase~=4)
xA = mvnrnd(muA,covA,N);
xB = mvnrnd(muB,covB,N);
else
    
xA = a;
xB = b;
end
figure('units','normalized','outerposition',[0 0 1 1])

figure(1);
subplot(1,3,1);
hold on
plot(xA(:,1),xA(:,2),'.r','DisplayName','Class A');
plot(xB(:,1),xB(:,2),'.b','DisplayName','Class B')

axis equal

%% Create sample test points to classify in the right class
boundary = [0 0];
MEDBoundary = [0 0];
GEDBoundary = [0 0];
MAPboundary = [0 0];

if (testCase~=4)

[X1 X2] = meshgrid(-10:0.5:10, -10:0.5:10);
else
[X1 X2] = meshgrid(0:10:450, 0:10:450);
end

%% Classify sample Test Points in one of the classes 
classifierType = 0;

while (classifierType < 2)
    
    for vectors = 1:length(X1)

        sampleData = [X1(vectors,:);X2(vectors,:)];

        if (classifierType == 0)

        %Minimum Euclidean Distance Classifier (MED)
        [ClassOutput,distance1,distance2] = euclidean_class(muA, muB,sampleData);
        elseif (classifierType == 1)

        %Generalized Euclidean Distance Classifier (GED)
        [ClassOutput,distance1,distance2] = mahalonobis_class(muA, muB, covA, covB, sampleData);
        end
        differencesDist = (distance1-distance2);
        boundaryIndices = find(differencesDist == 0);

        boundary = [boundary;[sampleData(:,boundaryIndices)]'];

        %% Plotting the newly classified points
        subplot(1,3,1);
        class1Indices = find(ClassOutput == 1);
        plot(sampleData(1,class1Indices),sampleData(2,class1Indices),'.g','DisplayName','Class A');
        class2Indices = find(ClassOutput == 2);
        plot(sampleData(1,class2Indices),sampleData(2,class2Indices),'.m','DisplayName','Class B');
        legend('show','Location','northwest')
        %figure(1),
        %axis([-10 10 -10 10]);

    end
        if classifierType == 0
            MEDBoundary = boundary;
            boundary = [0 0];
            classifierType = classifierType+1;
         
        elseif classifierType == 1
            GEDBoundary = boundary;
            classifierType = classifierType+1;
        end
        
       
end

%% Plotting the boundaries of MED and GED

%plot(boundary(2:end,1),boundary(2:end,2),'k');
subplot(1,3,1);
plot(MEDBoundary(2:end,1),MEDBoundary(2:end,2),'b');
%text(MEDBoundary(3,1),MEDBoundary(3,2),'\leftarrow MED Boundary','HorizontalAlignment','left');
plot(GEDBoundary(2:end,1),GEDBoundary(2:end,2),'k');
%text(GEDBoundary(3,1),GEDBoundary(3,2),'GED Boundary \rightarrow','HorizontalAlignment','right');
if (testCase~=4)

axis([-10 10 -10 10]);
else
    axis([0 450 0 450])
end
title('MED and GED Classifiers');
hold off

%% MAP Classifier
subplot(1,3,2);
hold on;
plot(xA(:,1),xA(:,2),'.r','DisplayName','Class A');
plot(xB(:,1),xB(:,2),'.b','DisplayName','Class B')

for vectors = 1:length(X1)

        sampleData = [X1(vectors,:);X2(vectors,:)];
        
        %MAP Classification
        pc = bayes_class(m,S,pw,sampleData); %can specify the number of classes or read from the variables
        
        differencesProb = pc(:,1)-pc(:,2);
        
        MAPboundaryIndices = find(differencesProb == 0);

        MAPboundary = [MAPboundary;[sampleData(:,MAPboundaryIndices)]'];

        %% Plotting the newly classified points
        subplot(1,3,2);
        axis equal;
        hold on;
        class1MAPIndices = find(differencesProb > 0);
        %class1Indices = find(differencesDist < 0);
        plot(sampleData(1,class1MAPIndices),sampleData(2,class1MAPIndices),'.g','DisplayName','Class A');
        class2MAPIndices = find(differencesProb < 0);
        plot(sampleData(1,class2MAPIndices),sampleData(2,class2MAPIndices),'.m','DisplayName','Class B');
        plot(MAPboundary(2:end,1),MAPboundary(2:end,2),'k','DisplayName','MAP');
        title(strcat('MAP classifier'));
        if (testCase~=4)

        axis([-10 10 -10 10]);
        else
            axis([0 450 0 450])
        end
        legend('show','Location','northwest')
        %axis equal
        %hold off;

    end

%% Plotting the Contours

subplot(1,3,3);
%plot(boundary(2:end,1),boundary(2:end,2),'k');
plot(MEDBoundary(2:end,1),MEDBoundary(2:end,2),'b');
hold on
plot(GEDBoundary(2:end,1),GEDBoundary(2:end,2),'k');


X3A = mvnpdf([X1(:),X2(:)],muA',covA);
X3B = mvnpdf([X1(:),X2(:)],muB',covB);
X3A = reshape(X3A,length(X2),length(X1));
X3B = reshape(X3B,length(X2),length(X1));

%surf(X1,X2,X3A);
contour(X1,X2,X3A);
%text(MEDBoundary(3,1),MEDBoundary(3,2),'\leftarrow MED Boundary','HorizontalAlignment','left');
contour(X1,X2,X3B);
%text(GEDBoundary(3,1),GEDBoundary(3,2),'GED Boundary \rightarrow','HorizontalAlignment','right');
title('Contour Plot');
if (testCase~=4)

axis([-10 10 -10 10]);
else
    axis([0 450 0 450])
end
axis equal

saveas(gcf,strcat('MED_GED_MAP_Case_',num2str(testCase),'.png'));
