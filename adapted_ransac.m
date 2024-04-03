% In this toy example:
% We consider two variables (features): variable 1 and variable 2,along with two classes, C1 and C2.
% For each variable, we generate two samples of size 624 from normal distributions that overlap.
% The data is then divided into 27 groups (equivalent to subjects).
% We subsequently introduce label noise through the following process:
% Given a probability p, for each subject and each sample label (0 or 1),
% we randomly draw a value x from the range 0 to 1.
% If x < p, we change the sample label.
% The first 5 subjects have a noise probability of 0.05.
% The second set of 5 subjects have a noise probability of 0.25.
% The third set of 10 subjects have a noise probability of 0.3.
% The last set of 7 subjects have a noise probability of 0.5.
% Following this, we apply our adapted RANSAC algorithm with a subset size set to 2 (n=2) for model fitting.
% We set the maximum number of iterations to 100, allowing for 100 random combinations of 2 subjects out of 27.
% This approach enables us to identify the atypical subjects.
clear; close all; clc;
%%%%%%%DATA generation%%%%%
N0=624;
N1=624;
%%%%%variable 1
pd=makedist('Normal','mu',200,'sigma',50);
D11= random(pd,N1,1);
pd=makedist('Normal','mu',400,'sigma',50);
D01= random(pd,N0,1);    
% figure; histogram(D11); hold on; histogram(D01);
var1=[D11;D01];
%%%%%variable 2
pd=makedist('Normal','mu',100,'sigma',50);
D12= random(pd,N1,1);
pd=makedist('Normal','mu',250,'sigma',50);
D02= random(pd,N0,1); 
% figure; histogram(D12); hold on; histogram(D02);
var2=[D12;D02];
D=[var1,var2];
%%%%1->label=1; 0->label=0
LABELALL=[ones(N1,1);zeros(N0,1)];
%%%%%noising label process%%%%
NewPart = struct('partitionindicesALL',{});
Np=27;
v=[1,0];
for i=1:2
    l=v(i);
    indicelab=find(LABELALL==l);
    Llab=length(indicelab);
    partitionindices = partition(Np,Llab);
    NewPart(i).partitionindicesALL=indicelab(partitionindices);
end
NewSUBJECTSP = struct('subjectsINDICES',{});
Sall=[]; LabelBruite=[]; indiceB=[];
probB=[0.05*ones(1,5),0.2*ones(1,5),0.3*ones(1,10),0.5*ones(1,7)];
for j=1:Np
    v=[];
    for i=1:2
        x=NewPart(i).partitionindicesALL;
        v=[v,x(j,:)];
    end
    NewSUBJECTSP(j).subjectsINDICES=v;
    %%%%%%labels initiaux
    valSubject=LABELALL(v);
    %%%%%labels bruités
    TB= labelnoise(probB(j),valSubject,1,0);
    %%
    LabelBruite=[LabelBruite;TB];
    %%indices de tous les sujets
    indiceB=[indiceB,v];
    Sall=[Sall;j*ones(length(v),1)];
end
D=D(indiceB,:);
LABELALL=LabelBruite;
label=[LABELALL,1-LABELALL];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SId=1:Np;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%ALGORITHM 1: ADAPTED RANSAC: n=2, tmax=1, tmax=100, TauACC=0.7,nmin=2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=2;tmax=1;TauACC=0.7;nmin=2;
%%%Number of participants:
allcomb=nchoosek(SId,n);
itmax=length(allcomb);
THETAMI = struct('network',{},'inliers',{});
THETAMODELS = struct('network',{},'inliers',{});
SI=[];NSI=0;
comb = randperm(length(allcomb));
for it=1:100
    it
    %%%%%%S is the set of n participants randomly selected
    S=allcomb(comb(it),:);
    %%%%%DATA Associated to S
    TrainDB=[];TrainLabel=[]; TrainSall=[];
    for i=1:length(S)
        Dxx=D(Sall==S(i),:);
        TrainDB=[TrainDB;Dxx];
        Lxx=label(Sall==S(i),:);
        TrainLabel=[TrainLabel;Lxx];
        Sxx=Sall(Sall==S(i),:);
        TrainSall=[TrainSall;Sxx];
    end
    %%%%LINES PAERMUATION%%%%%%%%%%
    [r, ~] = size(TrainDB);
    shuffledRow = randperm(r);
    TrainDB = TrainDB (shuffledRow, :);
    TrainLabel = TrainLabel (shuffledRow, :);
    TrainSall = TrainSall (shuffledRow, :);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nit=0;
    for t=1:tmax
        %%%%%%%%%%%%%%%%%%%%%%%%%%%ANN IMPLEMENTATION%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Use of the pattern recognition network 'patternet' from the Deep Learning Toolbox %%%%%%%%%%%%%%
        trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
        % Create a Pattern Recognition Network
        hiddenLayerSize = [6,4];
        net = patternnet(hiddenLayerSize, trainFcn);
        %%%USE of the ReLU activation function (poslin in matlab) rather than the Sigmoid function%%%%%%%%%%%%%%
        % net.layers{1}.transferFcn = 'poslin';
        % net.layers{2}.transferFcn = 'poslin';
        % view(net)
        net.divideParam.trainRatio = 100/100;
        net.divideParam.valRatio = 0/100;
        net.divideParam.testRatio = 0/100;
        net.trainParam.epochs=1000;
        net.trainParam.showWindow = 0;
        net = init(net);
        %%%%TRAINING OF THE NETWORK
        [net,~] = train(net,TrainDB',TrainLabel');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%ALGORITHME 2: DERIVE THE SUBSET OF INLIERS INLINE WITH THE MODEL%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Inl= algo2(net,Sall,D,TauACC,label,SId);
        %%%%%%%%%%%%%%%%%%%%%%%%%END OF ALGORITHM2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cardI=length(Inl);
        if (cardI>=nit)
            %%%%%%Save data associated to inlier participants and initialization parameters
            THETAMI(it).network=net;
            THETAMI(it).inliers=SId(Inl);
            nit=cardI;
        end
        
    end
    %%%%%%%%model (THETAMI(it).network) trained using (THETAMI(it).inliers) data
    %%%%%%Sit is the set of inliers THETAMI(it).inliers
    Sit=THETAMI(it).inliers;
    if (length(Sit)>=nmin)
        %%%%%DATA Associated to S
        TrainDB=[];TrainLabel=[]; TrainSall=[];
        for i=1:length(Sit)
            %%%%metrice de données%%%%%
            Dxx=D(Sall==Sit(i),:);
            TrainDB=[TrainDB;Dxx];
            Lxx=label(Sall==Sit(i),:);
            TrainLabel=[TrainLabel;Lxx];
            Sxx=Sall(Sall==Sit(i),:);
            TrainSall=[TrainSall;Sxx];
        end
        %%%%LINES PAERMUATION%%%%%%%%%%
        [r, c] = size(TrainDB);
        shuffledRow = randperm(r);
        TrainDB = TrainDB (shuffledRow, :);
        TrainLabel = TrainLabel (shuffledRow, :);
        TrainSall = TrainSall (shuffledRow, :);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        net=THETAMI(it).network; 
        net.trainParam.showWindow = 0;
        [net,tr]=train(net,TrainDB',TrainLabel');
        %%%%%%%%%%%%%ALGORITHM 2%%%%%%%%%%%%%%%%%%
        Inlit= algo2(net,Sall,D,TauACC,label,SId);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cardIit=length(Inlit);
        if (cardIit>=nmin)
        %%%%%Save data associated to inlier participants and the model
           THETAMODELS(it).network=net;
           THETAMODELS(it).inliers=SId(Inlit);
           SI=[SI,THETAMODELS(it).inliers];
           NSI=NSI+1;
        end
    end   
end
% savefile = 'OUTPUT_ALGO1.mat';
% save(savefile,'THETAMI','THETAMODELS');
%%%%%%%%%%%%%%%%%%%%%%ALGORIHM 3:Participant Clustering%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; histogram(SI,1:Np); ylabel('counts'), xlabel('subject index');
tauTOP=0.7; tauOUT=0.1;
scorei=zeros(1,Np);
for i=1:Np
    scorei(i)=sum(SI==SId(i))/NSI;
end
figure; plot(scorei,'*'); xlabel('subject index'); ylabel('frequency');
ITOP= SId(scorei>=tauTOP);
IOUT= SId(scorei<=tauOUT);
Inter=setdiff(SId,[ITOP,IOUT]);
% savefile = 'OUTPUT_ALGO3.mat';
% save(savefile,'ITOP','Inter','IOUT');