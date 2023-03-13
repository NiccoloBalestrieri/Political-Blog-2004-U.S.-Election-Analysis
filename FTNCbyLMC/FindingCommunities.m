%FINDING communities in a (possibly) DIRECTED, WEIGHTED network.
%
%Please cite: 
%C. Piccardi, Finding and testing network communities by
%lumped Markov chains, PLoS ONE, 6(11), e27028, 2011, 
%http://dx.doi.org/10.1371/journal.pone.0027028
%
%Copyright: 2011, Carlo Piccardi, Politecnico di Milano, Italy
%email carlo.piccardi@polimi.it
%
%Last updated: Nov 7, 2011
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%INPUT: 
%The file 
%   A_{netname}.mat 
%must be in the working directory and must contain the following variables
%in Matlab binary format:
%   i) A : NxN weight matrix defining the (strongly connected) network. 
%       A(i,j) is the weight of the link i->j.
%       If all the (nonzero) weights are 1, the network is actually UNWEIGHTED.
%       If A is symmetric, the network is actually UNDIRECTED.
%   ii) labels : (optional) 1xN cell vector of node labels (e.g., names)
%
%OUTPUT: 
%After the user has selected, among the proposed partitions, the
%one with q clusters:
%   i) the list of nodes of each community, with the associated persistence
%   probability, is printed on the screen;
%   ii) the Matlab binary file
%      com_{q}_{netname}.mat
%   is saved in the working directory. It contains the description of the 
%   partition, i.e., a N-dimensional vector com such that com(i) is the 
%   index of the community node i belongs to.
%
%PARAMETERS: 
%Please set the values of "Tsup", "maxcl", and "netname" in the
%section below.

clear all
close all
set(0,'Units','pixels') 
scn = get(0,'ScreenSize');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%SETTING PARAMETERS

%%%%%range to be explored for the time horizon T of the random walker
Tsup=20; %Tmin is always 1

%%%%%maximum number of clusters of the partitions
maxcl=10;

%%%%%name of the network: the file A_{netname}.mat will be loaded
%%%%%UNCOMMENT the name of the network to be loaded

%-----the following networks are analyzed in the PLoS ONE paper
%-----(see reference above): the datafiles are available in the 
%-----distribution package 

%-----these networks are analyzed in the main text of the paper
netname='blogs_scc';                  %set Tsup=20, maxcl=12
% netname='LFRuu_beta1mu025';       %set Tsup=15, maxcl=50
% netname='LFRuu_beta2mu06';        %set Tsup=10, maxcl=60
% netname='netscience_gc';          %set Tsup=10, maxcl=50
% netname='wtn2008_gc';             %set Tsup=10, maxcl=10
%-----these networks are analyzed in the supporting information text
% netname='ErdosRenyi';             %set Tsup=10, maxcl=20
% netname='Zachary';                %set Tsup=10, maxcl=10
% netname='LFdw_mu03';              %set Tsup=15, maxcl=50
% netname='LFdw_mu06';              %set Tsup=15, maxcl=50
% netname='linkrank';               %set Tsup=10, maxcl=10
% netname='neural_gc';              %set Tsup=10, maxcl=20

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%LOADING DATA, AND COMPUTING BASIC STATISTICS

disp([' '])
disp(['COMMUNITY ANALYSIS'])

%loads the NxN network matrix A 
%and (optionally) a Nx1 cell "labels" containing label strings
load(strcat('A_',netname,'.mat'));
A=full(A);

%if labels do not exist in the uploaded file,
%creates fictitious labels which are simply the node numbers
if length(find(char(who('*'))=='b'))==0 %labels do not exists in the file uploaded
    labels=cell(length(A),1);
    for i=1:length(A)
        labels(i)=cellstr(num2str(i));
    end;
end;

disp(['Network: ',netname,' - N = ',int2str(length(A))])
disp(['Computing the Markov matrix...'])

k_in=sum(A); %row vector of node in-weights (or in-degrees)
k_out=sum(A')'; %column vector of node out-weights (or out-degrees)
m=sum(k_in); %total weight (or total number of links) in the network
N=length(k_in); %number of nodes

%creating the Markov matrix by row-normalizing A
P=zeros(N,N);
rowsum=zeros(N,1);
for i=1:N
    rowsum(i)=sum(A(i,1:N));
    for j=1:N
        P(i,j)=A(i,j)/rowsum(i);
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%FINDING THE BEST TIME HORIZON T

disp(['Finding the best time horizon T ...'])
figure('OuterPosition',[1 2*scn(4)/3 2*scn(3)/3 scn(4)/3])
f1=get(0,'CurrentFigure');

cccvec=zeros(Tsup,1); %cccvec(T) is the cophenetic correlation coefficient
                      %when T is the time horizon

%cycling over various T
for T=1:Tsup
    %creating the similarity matrix S
    S=zeros(N,N);
    Ptot=P;
    Pcurr=P;
    for t=2:T
        if T>20 & rem(t,10)==0
            disp(['          step t = ',int2str(t),' of ',int2str(T)])
        end;
        Pcurr=P*Pcurr;
        Ptot=Ptot+Pcurr;
    end;
    S=(Ptot+Ptot')/T;
    for i=1:N %the diagonal terms are set to zero
        S(i,i)=0;
    end;

    %creating the distance matrix D by normalizing S
    D=zeros(N,N);
    minS=min(squareform(S));
    maxS=max(squareform(S));
    for i=1:N
        D(i,i)=0;
        for j=i+1:N
            D(i,j)=1-(S(i,j)-minS)/(maxS-minS);
        end;
    end;
    D=D+D';

    %creating the cluster tree (dendrogram)
    dlist=squareform(D);
    Z=linkage(dlist,'average');
    
    figure(f1)
    [alpha,beta,perm]=dendrogram(Z,0,'labels',labels,'colorthreshold','default');
    drawnow
    
    %computing the cophenetic correlation coefficient cccvec(T)
    [ccc,coph]=cophenet(Z,dlist);
    cccvec(T)=ccc;
    disp(['     T = ',int2str(T),'   C = ',num2str(ccc)])

end; %cycling over various T

%selecting the best time horizon T
figure('OuterPosition',[1 4*scn(4)/9 scn(3)/4 2*scn(4)/9])
f2=get(0,'CurrentFigure');
figure(f2)
plot(1:Tsup,cccvec,'ko-')
drawnow
ylabel('coph. corr. coeff. \it{C}')
xlabel('time horizon \it{T}')
grid on

[cccmax,Tmax]=max(cccvec);
disp(['Maximum cophenetic correlation coefficient C = ',num2str(cccmax),' at T = ',int2str(Tmax),' - Recomputing cluster tree...'])

%REPEATING CLUSTER ANALYSIS WITH T=Tmax

T=Tmax;
%creating the similarity matrix S
S=zeros(N,N);
Ptot=full(P);
Pcurr=full(P);
for t=2:T
    if T>20 & rem(t,10)==0
        disp(['          step t = ',int2str(t),' of ',int2str(T)])
    end;
    Pcurr=P*Pcurr;
    Ptot=Ptot+Pcurr;
end;
S=(Ptot+Ptot')/T;
for i=1:N %the diagonal terms are set to zero
    S(i,i)=0;
end;

%creating the distance matrix D by normalizing S
D=zeros(N,N);
minS=min(squareform(S));
maxS=max(squareform(S));
for i=1:N
    D(i,i)=0;
    for j=i+1:N
        D(i,j)=1-(S(i,j)-minS)/(maxS-minS);
    end;
end;
D=D+D';

%creating the cluster tree (dendrogram)
dlist=squareform(D);
Z=linkage(dlist,'average');
figure(f1)
[alpha,beta,perm]=dendrogram(Z,0,'labels',labels,'colorthreshold','default');
drawnow
ylabel('distance')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%FINDING PARTITIONS WITH 2 TO maxcl CLUSTERS

disp(['Finding partitions with 2 to ',int2str(maxcl),' clusters...'])

%computing Markov asymptotic distribution (x)
AAA=eye(N)-P'; 
AAA(N,:)=1;
bbb=zeros(N,1); 
bbb(N)=1;
x=AAA\bbb;

figure('OuterPosition',[1 1 4*scn(3)/9 4*scn(4)/9])
f3=get(0,'CurrentFigure');
figure(f3)

for q=2:maxcl
    
    com=cluster(Z,'maxclust',q);  %com(i) is the community of node i
    nc=max(com);                  %number of communities (=q, in principle)
    
    %H codes the partition
    H=zeros(N,nc);
    for i=1:N
        H(i,com(i))=1;
    end;
    
    %U is the lumped Markov matrix
    U=(diag(H'*x))^(-1)*H'*diag(x)*P*H;
    
    %creating the persistence probabilities' diagram
    plot(nc*ones(nc,1),diag(U),'kx')
    hold on
    plot([nc nc],[min(diag(U)) max(diag(U))],'k');

    disp(['     q = ',int2str(nc),' - min u_cc = ',num2str(min(diag(U)))])

end;
axis([1 maxcl+1 0 1])
ylabel('persistence probs. \it{u_{cc}}')
xlabel('number of communities \it{q}')
grid on

%selecting and saving one or more partitions
while 1 %stopping only with ctrl-c
    q_input=input('Which partition do you want to save (or ctrl-c to stop) ? q = ');
    com=cluster(Z,'maxclust',q_input);
    nc=max(com);                  %number of communities (=q_input, in principle)
    %H codes the partition
    H=zeros(N,nc);
    for i=1:N
        H(i,com(i))=1;
    end;
    %U is the lumped Markov matrix
    U=(diag(H'*x))^(-1)*H'*diag(x)*P*H;

    savename=strcat('com_',int2str(q_input),'_',netname,'.mat');
    save(savename,'com')

    %computing modularity matrix for the saved partition
    Amod=(A-k_out*k_in/m)/m;
    %computing modularity
    Q4=0;
    for c=1:q_input
        Q4=Q4+sum(sum(Amod(com==c,com==c)));
    end;
    disp(['Partition with q = ',int2str(q_input),' saved in ',savename,' - modularity Q = ',num2str(Q4)])
    
    %displaying communities and persistence probabilities on the screen
    for c=1:nc
        disp(['Community # ',int2str(c)])         
        disp(['number of nodes = ',int2str(length(find(com==c)))])
        disp(['persistence probability = ',num2str(U(c,c))])
        listc=labels(com==c);
        disp(['list of nodes:'])
        disp(listc)
        disp([' ']);
    end;
        
end;
