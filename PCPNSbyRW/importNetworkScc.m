data = readtable("C:\Users\nicco\OneDrive\Documenti\GitHub\Political-Blog-2004-U.S.-Election-Analysis\PCPNSbyRW\largest_componentA.csv");
% Estrai i nodi e gli archi dal dataframe

nodes = unique([data.Source; data.Target]);
edges = table2array(data(:, [1, 2]));

%weights = data.Weight;

G = digraph(edges(:, 1), edges(:, 2));

num_nodes = numnodes(G);
num_edges = numedges(G);

display(G.Nodes);
A = adjacency(G);

save('A_blogs_scc.mat', 'A');
