C. Piccardi, 
FINDING AND TESTING NETWORK COMMUNITIES BY LUMPED MARKOV CHAINs, 
PLoS ONE, 6(11), e27028, 2011, 
http://dx.doi.org/10.1371/journal.pone.0027028

Copyright: 2011, Carlo Piccardi, Politecnico di Milano, Italy
email carlo.piccardi@polimi.it

Last updated: Nov 7, 2011


%%%%%%%%%%%%%%%%%%%%%%%%%%%%


The distribution file FTCbyLMC.zip contains the Matlab codes and the datafiles used to produce the results and the figures of the above paper (both in the main text and in the supplementary information). Please refer to the paper for all details about the algorithms and the data.

The zip file contains:

  FindingCommunities.m : 

fully automated community detection - computes the node distances, performs hierarchical analysis, generates a set of meaningful partitions and, for each one of them, computes the set of persistence probabilities.

  TestingCommunities.m :

computes the persistence probabilities of a single, "a priori" given partition.


The following datafiles are included:

  A_toy12.mat : the toy network of Figs. 1 and 2.
  A_LFRuu_beta1mu025.mat : undirected, unweighed LFR benchmark (Fig. 4, top-left panel).
  A_LFRuu_beta2mu06.mat : undirected, unweighed LFR benchmark (Fig. 5, top-left panel).
  A_netscience_gc.mat : giant component of the Netscience network (Fig. 6).
  A_wtn2008_gc.mat : giant component of the world trade network in 2008 (Fig. 8).

  A_ErdosRenyi.mat : an instance of an Erdos-Renyi network (Fig. S1.1).
  A_Zachary.mat : Zachary's karate club network (Fig. S1.2).
  A_LFdw_mu03.mat : directed, weighted LFR benchmark (Fig. S1.3, top panel).
  A_LFdw_mu06.mat : directed, weighted LFR benchmark (Fig. S1.3, bottom panel).
  A_linkrank.mat : LinkRank benchmark network (Fig. S1.5).
  A_neural_gc : giant component of the neural network of the Caenorhabditis elegans (Fig. S1.5).

  com_3_toy12.mat : an example of a file describing a partition, to be used as input for TestingCommunities.m. It contains the "natural" 3-cluster partition of the net A_toy12.mat (Figs. 1 and 2).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SOFTWARE REQUIREMENTS

The .m codes work correctly with Matlab 7.11.0 operating under Windows 7 operating system (but they probably work on previous versions as well).

FindingCommunies.m requires the availability of the Statistics Toolbox for the hierarchical analysis.
