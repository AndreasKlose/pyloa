"""
    Module net.net_prob of package pyloa:
    Base class for a network location problem.
"""
import numpy as np
from os.path import basename
from timeit import time
from pyloa.net.parser import read_graph 
from pyloa.mip.model import set_mipSolver

#--------------------------------------------------------------------------

class NetProblem:
    """
    Base class for a network location problem.
    """
    
    def __init__( self, fname=None, dstmat='matrix', orlib=False, d=None, 
                  dmax=None, m=0, n=0, w=None, p=None ):
        """
        Creates an instance of a network location problem.
        
        Parameters
        ----------   
        * fname : str, optional  
            None or name of the input file or problem instance to be solved.
            If None, it is assumed that the data are provided via arguments
            d, w, and p. Otherwise, fname is either the name (including the
            path) of a text file keeping the data or the problem instance
            to be taken from Beasley's OR Library. Default: None  
        * dstmat : str  
            This argument is only relevant if fname is not None. It specifies
            how the distances should be stored. Possible values are 'matrix',
            'dict' or 'vector'. See the parameter 'd' for further explanations.  
        * orlib : bool, optional  
            True if the data file originates from Beasley's OR Library.
            Default: False  
        * d : None or numpy array of int/float or dictionary of int/float or a 
            function  
            If fname is None, then d need to keep the distances between
            the m nodes of the graph and the n <= m nodes where facilities
            can be established.  
            1. If the set of facility nodes and customer nodes is identical
               and thus the matrix of distances symmetric, d can be one
               of the following:  
                1. d is a numpy (mxn), n=m, array of int or float and
                    d[i,j]=d[j,i] the distance between nodes i and j,
                    d[i,i]=0.  
                2. d is a dictionary of length m*(m-1)//2 and d[(i,j)] gives 
                    the distance between nodes i and j for i=1,...,m-1 and 
                    j = 0,..., i-1.  
                3. d is a numpy array of size m*(m-1)//2. The distance
                    between nodes i (i=1,...,m-1) and j (j=0,...,i-1) must
                    then be stored in d[ i*(i-1)//2 + j ].  
                4. d is a function, so that d(i,j) returns the distance
                    beween nodes i (i=1,...,m-1) and j (j=0,...,i-1).  
            2. Set of customer and facility nodes are not identical.  
               In this case d need to be a numpy mxn matrix just as
               under (1.1).  
        * dmax : None or numpy array of int or float  
            dmax[i] specifies the maximal distance for customer node i
            in maximal covering location           
        * m : int  
            Number of customers nodes. Need only to be specified if d
            is a function!!!  
        * n : int  
            Number of facility nodes (n <= m). Need only to be specified if
            d is a function!!!  
        * w : None or a numpy array of positive int  
            If None, all weights are set to one. Otherwise, w must be a numpy 
            array of int and w[i] is the weight of node i=0,...,m-1.  
        * p : None or int  
            The number of facilities to locate 
        """
        self.prob_name = None if fname is None else basename(fname).split('.')[0]
        """Name of data file or None"""
        
        self._p = 0 if p is None else p 
        """Number of facilities to locate on the graph"""
        
        self._m = 0 
        """Number of (customer) vertices of the graph"""
        
        self._n = 0
        """Number of possible facility vertices of the graph"""
        
        self._w = None 
        """Weights of the m customer nodes"""
        
        self._dmat = None  
        """Determines the distance between customer and facility nodes"""

        self._dmax = dmax
        """Maximal distances to be observed in maximal covering location"""
               
        self._max_wdist = None 
        """Will equal the largest weighted distance for each vertice i"""
        
        self.min_wdist = None 
        """Function returning a numpy array that gives for each customer
           node its weighted distance to the nearest facility from 
           a given list S of established facilities"""
 
        self.get_coverage = None 
        """Function return amount of demand covered by a list S of open facilities"""
        
        self._dst_type = None 
        """Data type of the single distances, i.e. float or int"""
               
        self.__model = None
        """Model of an underlying integer programming problem"""
        
        self.__y = None 
        """Binary location variables in MIP model"""
        
        self.__x = None 
        """Allocation variables in MIP model"""
        
        self.__z = None 
        """Covering or Elloumi z-variables in MIP model"""
        
        self.__silent = False 
        """If False, solvers will send log-output to stdout"""
        
        self.facilities = None 
        """List of facility node indices in a solution"""
        
        self.assigned = None 
        """assigned[i] gives the nearest facility node to node i in a solution"""
        
        self.get_assigned = None 
        """Function returning as a list that for each customer node gives
           the nearest facility from a given set S of facilities"""
            
        self.wdist = None 
        """Total weighted distance in a solution"""
        
        self.bound = 0
        """Lower (or upper) bound on optimal objective value"""
        
        self.itr = 0
        """Any iteration counter"""

        self.__stime = (0.0,0.0)
        """Start time of a solution procedure"""
        
        self.ctime = (0.0,0.0)
        """Computation time (CPU and Walltime) required to solve a problem instance""" 
        
        self.mip_time = 0.0
        """Computation time (walltime seconds) spent in the MIP solver"""
        
        self.mip_work = 0
        """Deterministic computation time. For Cplex this is the total amout
        of CPU ticks. For GuRoBi one work unit is about one second on single
        thread."""
        
        self.nodeCount = 0
        """Number of nodes processed in total when solving with a MIP solver""" 
        
        self._is_matrix = False 
        
        if fname is None:  
            if not d is None: self.set_data(d, self._p, w=w, dmax=dmax, m=m, n=n)                    
        else: 
            p, d, w, dmax = read_graph( fname, dmatrix=dstmat, orlib=orlib )
            self.set_data( d, p, w=w, dmax=dmax )
    
    #---------------------------------------------
    
    def reset( self ):
        """
        Nullify the problem's data, solution values, etc.
        """
        self.prob_name = None 
        self._p = 0 
        self._m = 0 
        self._n = 0
        self._w = None   
        self._dmat = None  
        self._dmax = 0
        self._max_wdist = None 
        self.min_wdist = None 
        self.get_coverage = None 
        self._dst_type = None
        if not self.__model is None: self.__model.end() 
        self.__model = None
        self.__y = None 
        self.__x = None 
        self.__z = None 
        self.facilities = None 
        self.assigned = None 
        self.get_assigned = None 
        self.wdist = None 
        self.bound = 0
        self.itr = 0
        self.__stime = (0.0,0.0)
        self.ctime = (0.0,0.0)
        self.mip_time = 0.0
        self.mip_work = 0
        self.nodeCount = 0
        self._is_matrix = False  
        
    #---------------------------------------------
   
    def set_data( self, d, p=1, w=None, dmax=None, m=None, n=None ): 
        """
        Pass data to this instance of a network location problem.
        
        Parameters  
        ----------   
        * d : Numpy array of int/float or dictionary of int/float or a function  
            d keeps the distances between the m nodes of the graph and the 
            n <= m nodes where facilities can be established. 
             
            1. If the set of facility nodes and customer nodes is identical
               and thus the matrix of distances symmetric, d can be one
               of the following:
               
               1. d is a numpy (mxn), n=m, array of int or float and
                  d[i,j]=d[j,i] the distance between nodes i and j,
                  d[i,i]=0.
               2. d is a dictionary of length m*(m-1)//2 and d[(i,j)] gives 
                  the distance between nodes i and j for i=1,...,m-1 and 
                  j = 0,..., i-1. 
               3. d is a numpy array of size m*(m-1)//2. The distance
                  between nodes i (i=1,...,m-1) and j (j=0,...,i-1) must
                  then be stored in d[ i*(i-1)//2 + j ]. 
               4. d is a function, so that d(i,j) returns the distance
                  beween nodes i (i=1,...,m-1) and j (j=0,...,i-1).  
            
            2. Set of customer and facility nodes are not identical.
               In this case d need to be a numpy mxn matrix just as
               under (1.1).
                     
        * p : int  
            The number of facilities to locate  
            
        * w : None or a numpy array of positive int  
            If None, all weights are set to one. Otherwise, w must be a numpy 
            array of int and w[i] is the weight of node i=0,...,m-1.  
            
        * dmax : None or numpy array of int or float  
            dmax[i] specifies the maximal distance for customer node i
            in maximal covering location  
            
        * m, n : int  
            In case that d is a function, m and n need to be specified.
            m gives the number of customer nodes and n the number of
            facility nodes. The function d should then return the 
            distance between customer node i and facility node j as 
            d(i,j).                                
        """
        self._dmax = dmax 
        self._is_matrix = False 
        if callable(d):
            self._d = d
            self._m = m 
            self._n = n  
        else:
            self._dmat = d  
            self._is_matrix = type(d)==np.ndarray and len(d.shape) > 1
            if self._is_matrix: 
                self._m, self._n = d.shape
                self._d = lambda i,j : self._dmat[i,j] 
            else: 
                self._n = int( np.sqrt( 2*len(d)+0.25 ) + 0.51 )
                self._m = self._n
                self._d = self.__dict_dist if type(d)==dict else self.__sparse_dist
        
        self._p = max(1, min( p, self._n ) ) 
        self._w = np.ones(self._m, dtype=int) if w is None else w
        self._dst_type = type( self._d(1,0) ) 
        if self._is_matrix:
            self._max_wdist = self._w*np.max( self._dmat, axis=1 )
            self.min_wdist = lambda S : self._w * self._dmat[:,S].min(axis=1)
            self.get_assigned = lambda S : list(map(lambda i : S[i], self._dmat[:,S].argmin(axis=1)) )
            self.get_coverage = lambda S : 0 if self._dmax is None else \
                                           np.dot(self._w, self._dmat[:,S].min(axis=1) <= self._dmax)
            self.get_radius = lambda S : self.min_wdist(S).max()                                
        else:    
            # This case assumes a symmetric matrix of distances between customer and facility nodes
            self._max_wdist = np.fromiter( (self._w[i]*max(self._d(i,j) for j in range(self._n)) \
                                            for i in range(self._m)), dtype=self._dst_type ) 
            self.min_wdist = lambda S : np.fromiter( (self._w[i]*min(self._d(i,j) for j in S) \
                                            for i in range(self._m)), dtype=self._dst_type ) 
            self.get_assigned = lambda S : list(map(lambda i : min( S, key = lambda j : \
                                                self._d(i,j) ), range(self._m) ))
            self.get_radius = lambda S : self.min_dist(S).max()
            self.get_coverage = lambda S : 0 if self._dmax is None else \
                    sum( self._w[i]*any(self._d(i,j) <= self._dmax[i] for j in S) for i in range(self._m) )
              
    #---------------------------------------------
    
    def __dict_dist(self, i, j ):
        """ Return distance between i and j"""
        if i==j: return 0 
        return self._dmat[(i,j)] if i > j else self._dmat[(j,i)]
    
    #---------------------------------------------
    
    def __sym_dist(self, i, j):
        if i==j: return 0
        return self._dmat[i*(i-1)//2+j] if i > j else self._dmat[j*(j-1)//2 + i]

    #---------------------------------------------

    def _set_starttime( self ):
        """
        Initialize start time of a solving process.
        """
        self.__stime = (time.process_time(),time.time())
        self.itr = 0
        
    #---------------------------------------------

    def _set_comptime( self, mip_time=True ):
        """
        Determine time required for solving a problem.
        If mip_time is True, it is assumed that the
        MIP solver did the computation and its computation 
        time is accessed.
        """ 
        self.ctime = (time.process_time()-self.__stime[0],time.time()-self.__stime[1])
        if mip_time and not self.__model is None:
            self.mip_time = self.__model.runtime
            self.mip_work = self.__model.dettime 
            self.nodeCount = self.__model.nodeCount
        
    #---------------------------------------------
    
    @property 
    def mipSolver(self):
        """
        Return the MIP solver's name used for solving MIPs.
        """
        return set_mipSolver( )
    
    @mipSolver.setter 
    def mipSolver(self, solver ):
        set_mipSolver(solver)
        
    @property 
    def model(self):
        """MIP model of the problem instance"""
        return self.__model 
    
    @model.getter 
    def model(self):
        """Handle to the MIP model""" 
        return self.__model 
    
    @model.setter 
    def model(self, value ):
        """Handle to the MIP model""" 
        self.__model = value 
    
    @model.deleter 
    def model(self): 
        if not self.__model is None: self.__model.end() 
        self.__model = None 
        
    @property 
    def x(self):
        """
        Demand allocation variables x[i,j], where i is a 
        customer and j a facility node. 
        """
        return self.__x
    
    @x.setter
    def x(self, value ):
        """Demand allocation variables x[i,j], where i is a 
        customer and j a facility node.""" 
        self.__x = value  
    
    @x.getter 
    def x(self): 
        """Demand allocation variables x[i,j], where i is a 
        customer and j a facility node."""
        return self.__x     
    
    @property 
    def y(self):
        """
        Location binary variables y[j] for 
        j=0,...,n-1 and n number of facility nodes.
        """
        return self.__y
        
    @y.setter
    def y(self, value ):
        """
        Location binary variables y[j] for 
        j=0,...,n-1 and n number of facility nodes.
        """ 
        self.__y = value  
            
    @y.getter 
    def y(self):
        """
        Location binary variables y[j] for 
        j=0,...,n-1 and n number of facility nodes.
        """ 
        return self.__y    
        
    @property 
    def z(self):
        """
        Covering variables in maximal covering location
        or variables z in Elloumi formulation.
        """
        return self.__z
        
    @z.setter
    def z(self, value ):   
        """
        Covering variables in maximal covering location
        or variables z in Elloumi formulation.
        """
        self.__z = value
        
    @z.getter 
    def z(self): 
        """
        Covering variables in maximal covering location
        or variables z in Elloumi formulation.
        """
        return self.__z    
    
    @property
    def silent( self ):
        """
        If silent is True, solution methods do not give log-output
        """
        return self.__silent 
    
    @silent.setter
    def silent( self, value ):
        """
        Set property silent to True or False.
        """ 
        self.__silent = bool(value)