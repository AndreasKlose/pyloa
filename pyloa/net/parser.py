"""
    Read (graph) data of a network location problem.
"""
import numpy as np
import networkx as nx
import os
from urllib.request import urlretrieve
   
#----------------------------------------------------------------------

def _findEdge ( edge, lste ):
    """
    Returns index of edge "edge" in the list "lste" of edges
    and -1 if e is not in the list.
    """
    for i, e in enumerate( lste ):
        if edge[0]==e[0] and edge[1]==e[1]: return(i)
    return(-1)

#----------------------------------------------------------------------

def comp_dist_mat( G, dmatrix ):
    """
    Compute the length of shortest path between all nodes in G and
    return the result as a 2-dim numpy array. Note that the edges
    in the graph MUST have the attribute 'weight' which refers
    to the non-negative length of an edge.
    
    Parameters
    ----------
    G : networkx graph (undirected)
        G is a (networkx) graph. Each edge of the graph need to
        have the edge attribute 'weight' to which the non-negative 
        length of an edge is associated. Nodes of the graph are 
        numbered/identified as strings '1',...,'n', where n is 
        the number of nodes.
    dmatrix : str  
        Specifies how to return the distance matrix. dmatrix
        must be one of the following:  
        
        1. dmatrix = 'dict' : In this case a dictionary d
           is returned so that d[(i,j)] gives the distance
           between nodes i=1,...,n-1 and  j=0,...,i-1. 
           Note that this case requires the set of customer
           and facility nodes to be identical so that
           the matrix of distances is symmetric.  
        
        2. dmatrix = 'matrix' : In this case a nxn numpy
           array d is returned and d[i,j] is the distance
           between customer node i=0,...,n-1 and facility node
           j=0,...,n-1. In this case it is not required that 
           customer node i is identical to facility node j
           if i==j. The matrix of distances is thus not
           necessarily symmetric.  
        
        3. dmatrix = 'vector': In this case a numpy array d
           of dimension n*(n-1)//2 is returned so that
           d[ i*(i-1)//2 + j] gives the distance between
           node i=1,...,n-1 and node j=0,...,i-1.
           Again it is assumed that all customer nodes
           are also facility nodes, so that the matrix
           of distances is symmetric.
            
    Returns
    -------
    d : dict or numpy array 
        distances between nodes as explained above.            
    """
    n = G.number_of_nodes()
    spl = nx.shortest_path_length(G, weight='weight')
    dmat = dmatrix.lower()
    if dmat=='matrix':
        return np.array([[s[1][str(j+1)] for j in range(n)] for s in spl],dtype=int)
    elif dmat=='dict':
        d = dict()
        for s in spl:
            i = int(s[0])-1
            for j in range(i): d[(i,j)] = s[1][str(j+1)]
    else: 
        dim = n*(n-1)//2
        d = np.zeros(dim,dtype=int)
        for s in spl:
            i = int(s[0])-1
            for j in range(i): d[i*(i-1)//2+j] = s[1][str(j+1)]
    return d
        
#------------------------------------------------------------

def read_graph( fname, dmatrix = None, orlib=False ):
    """
    Reads data of a graph from a text file called 'fname'. The
    data in the graph need to structured as follows:
    
        number of nodes, number of edges, number p of facilities
        For every edge e: end node 1, end node 2, length
        for each node: weight of the node, covering distance
    
    Thereby, p is the number of facilities to establish.
    Data on the nodes' weight and covering distance is optional.
    If not supplied, weights are assumed to equal 1 and it
    is assumed that information on covering distances is not needed.
    If covering distances are irrelevant and thus also not included
    in the data file, it is possible to just supply weight data
    in case that node weights are different. If covering distances
    (as, e.g., for maximal covering location) are needed, both
    need to be supplied, node weights and the covering distance
    of a node, even if all weights are equal to 1.
    
    Parameters
    ----------
    fname : str
        (Path) and file name of the data file.
    dmatrix : None or str  
        If None, the underlying graph is read from file and
        returned as a networkx graph object. Otherwise,
        the length of the shortest path in the graph are
        computed and returned in a way specified by 'dmatrix'
        as follows:
        dmatrix = 'dict' : In this case a dictionary d
        is returned so that d[(i,j)] gives the distance
        between nodes i=1,...,n-1 and j=0,...,i-1.
        dmatrix = 'matrix' : In this case a nxn numpy
        array d is returned and d[i,j] is the distance
        between nodes i=0,...,n-1 and j=0,...,n-1,
        dmatrix = 'vector': In this case a numpy array d
        of dimension (n-2)*(n+1)//2 is returned so that
        d[ i*(i-1)//2 + j] gives the distance between
        node i=1,...,n-1 and node j=0,...,i-1.
    orlib : bool, optional
        If True, the data file is assumed to be from Beasley's
        OR library. The graph data does then not include
        weights of the nodes. Node weights are thus assumed
        to equal 1.  If the file cannot  be found locally, it 
        is tried to download the file from the OR-lib web page.
            
    Returns
    -------
    p : int
        Number of facilities to locate
    G or d : either the networkx graph object or matrix d 
        Note that the data files are structured as follows:
        number of nodes, number of edges, number p of facilities
        For every edge e: end node 1, end node 1, length
            
        The OR-lib data have a small flaw. Sometimes, an edge is
        listed twice. In this case, the last read edge is supposed
        to give the correct data. 
            
        If the data file cannot be found or downloaded, None is
        returned.  
    w : None or numpy array of int
        Weights of the nodes. If the file does not contain node weights
        None is returned.    
    dmax : None or numpy array of int
       Covering distance for each node if supplied in the data file.
    """
    
    if not os.path.isfile(fname) and orlib:
        # Try to get the file from ORlib
        try:
            url = 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/'+fname
            urlretrieve( url, fname )  
        except:
            if dmatrix: return 0, [None]*3
            return 0, None, None
    
    if not os.path.isfile(fname): 
        raise Exception("Cannot read or download file "+fname)
    
    f = open( fname, 'r' )
    # read first line of the file n=#nodes and m=#edges
    n, m, p = list( map(int, f.readline().split() ) )

    # read remaining lines of the file
    edges = []
    for line in f:
        l = line.split()
        e = (l[0],l[1],l[2]) if l[0] < l[1] else (l[1],l[0],l[2])
        pos = _findEdge( e,edges )
        if pos < 0: 
            edges.append( e )
        else:
            m -= 1
            edges[pos] = e
        if len(edges)==m: break
        
    w = list()    # node weights if supplied
    dmax = list() # covering distances if supplied
    for line in f:
        l = list( map(int, line.split()) )
        if len(l) > 0:
            w.append(l[0])
            if len(l) > 1: dmax.append(l[1])
            if len(w)==n: break 
    if len(w)==0: w = None
    if len(dmax)==0: dmax = None
        
    f.close()
        
    # set up the graph using the list of edges
    G = nx.Graph()
    for e in edges:
        G.add_edge( e[0], e[1], weight=int(e[2]) )
        
    if not w is None: w = np.array(w)
    if not dmax is None: dmax = np.array(dmax)
  
    if dmatrix: return p, comp_dist_mat(G, dmatrix), w, dmax
    return p, G, w

#------------------------------------------------------------

def read_orlib_graph( fname, dmatrix=None ):
    """
    This is just a short cut for read_graph(fname, dmatrix=..., orlib=True).
    However only two values are returned, the number p and the graph (or 
    distance matrix).
    """
    p, Gd, _, _ = read_graph(fname, dmatrix=dmatrix, orlib=True)
    return p, Gd
