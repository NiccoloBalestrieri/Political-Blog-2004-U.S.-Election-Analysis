%TESTING communities in a (possibly) DIRECTED, WEIGHTED network.
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
%A Matlab binary format file (.mat) containing the description of the 
%partition must also be in the working directory (no rules on its name). 
%It must contain a the description of the partition, i.e., a N-dimensional 
%vector com (the name of the variable is mandatory) such that com(i) is the 
%index of the community node i belongs to. The indeces of the communities 
%must run from 1 to %q (=number of communities), namely 1<=com(i)<=q.
%
%OUTPUT: 
%The list of nodes of each community, with the associated persistence
%probability, is printed on the screen.
%
%PARAMETERS: 
%Please set "netname" and the name of the partition file in the
%section below.

clear all
close all
set(0,'Units','pixels') 
scn = get(0,'ScreenSize');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%SETTING PARAMETERS

%%%%%name of the network: the file A_{netname}.mat will be loaded
%%%%%UNCOMMENT the name of the network to be loaded

%-----the following networks are analyzed in the PLoS ONE paper
%-----(see reference above): the datafiles are available in the 
%-----distribution package 

%-----these networks are analyzed in the main text of the paper
netname='toy12';                  
% netname='LFRuu_beta1mu025';       
% netname='LFRuu_beta2mu06';        
% netname='netscience_gc';          
% netname='wtn2008_gc';             
%-----these networks are analyzed in the supporting information text
% netname='ErdosRenyi';             
% netname='Zachary';                
% netname='LFdw_mu03';              
% netname='LFdw_mu06';              
% netname='linkrank';               
% netname='neural_gc';              

%%%%%name of the partition file:
partition_name='com_3_toy12.mat';

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

%loading the partition file
load(partition_name);
if length(com)~=N
    disp(['DATA ERROR: length of "com" not equal to N.'])
    break
end;
    
%now com(i) is the community of node i

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%COMPUTING PERSISTENCE PROBABILITIES

disp(['Computing persistence probabilities...'])

%computing Markov asymptotic distribution (x)
AAA=eye(N)-P'; 
AAA(N,:)=1;
bbb=zeros(N,1); 
bbb(N)=1;
x=AAA\bbb;

figure('OuterPosition',[1 1 4*scn(3)/9 4*scn(4)/9])
f1=get(0,'CurrentFigure');
figure(f1)

nc=max(com);
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

axis([nc-1 nc+1 0 1])
set(gca,'XTick',nc-1:1:nc+1)
ylabel('persistence probs. \it{u_{cc}}')
xlabel('number of communities \it{q}')
grid on

%displaying results on the screen
%computing modularity matrix for the saved partition
Amod=(A-k_out*k_in/m)/m;
%computing modularity
Q4=0;
for c=1:nc
    Q4=Q4+sum(sum(Amod(com==c,com==c)));
end;
disp(['Partition with q = ',int2str(nc),' - modularity Q = ',num2str(Q4)])
   
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
