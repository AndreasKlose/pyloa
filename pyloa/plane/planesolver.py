"""
    Implements a class for solving planar minisum or minimax
    location problems.
"""
import numpy as np 
from os.path import basename
from timeit import time
from pyloa.mip.model import set_mipSolver
from pyloa.util import lla2xy, euclid
from pyloa.plane.parser import read_points, return_lola
from pyloa.plane.weber import solveWeber
from pyloa.plane.center import elzinga_hearn, charalambous, welzl, growRadius
from pyloa.plane.mip import weber_mipq, pcenter_mipq, SOCPcenter, SOCPweber
from pyloa.plane.loc_alloc import locAlloc, pmedian_heuristic, weber_vns, twoFacility
from pyloa.plane.weber_cg import colgen, set_optTol 
from pyloa.plot import plot_points

#--------------------------------------------------------------------------

class PlaneSolver:
    """
    Class for addressing a planar location problem.
    """
    
    def __init__(self, fname=None, Y=None, geospatial=False, w=None, g_coo=None, names=None):
        """
        Creates an instance of a planar location problem.
        
        Parameters
        ----------   
        fname : str, optional  
            None or name of the input file or problem instance to be solved.
            If None, it is assumed that the data are provided via arguments
            Y and w. Otherwise, fname need to be the name (and path if 
            required) of a text/csv file keeping the data of the problem 
            instance. See the module plane.parser for information on how
            the data file need to look like.
        Y : None or a mx2 numpy array of float  
            This argument is only used in case that fname=None. If it is
            not None, then Y must contain in each row the coordinates of 
            the m customer locations.
        geospatial: bool, optional  
            If False (the default), it is assumed that the rows of Y give 
            coordinates in the Euclidean plane. Otherwise, it is assumed
            that Y[i,0] gives the longitude and Y[i,1] the latitude
            of customer point i. 
        w : None or numpy array of m float or int  
            This argument is only used if fname=None. In this case,
            it gives the array of positive customer weights if not None.
            If None but fname and Y aren't, the weights are set to 1 for 
            all customer points.
        g_coo : Tuple of two lists, optional
            Longitude and latitude data of the customer points if not
            Y already contains these data (geospatial=True).
        names : None or list/array of str, optional  
            Names of the customer points if not None.
        """
        self.prob_name = None if fname is None else basename(fname).split('.')[0]
        """Name of the data file or None."""
        
        self.__Y = None  
        """Array of customer coordinates in Euclidean plane."""
        
        self.__w = None  
        """Array of customer weights."""
        
        self.__X = None 
        """(px2) array of facility coordinates in a solution where p facilities
           had to be located."""
           
        self.__lo = None 
        """Longitude data of customer points if provided."""
        
        self.__la = None 
        """Latitude data of customer points if provided."""
        
        self.__names = None 
        """Names of customer points if provided."""
           
        self.__sum_wdist = None 
        """
        The total weighted sum of distances to nearest facility in a solution.
        """
        
        self.__max_wdist = None 
        """
        The maximal weighted distance to nearest facility in a solution.
        """
        
        self.__sum_dist = None 
        """
        The total (unweighted) sum of distances to the nearest facility in a solution.
        """
        
        self.__max_dist = None 
        """
        The maximal (unweighted) distance to nearest facility in a solution.
        """
        
        self.__lobnd = 0 
        """Any lower bound on optimal objective function value."""
        
        self.__assigned = None 
        """
        Gives for each customer point the index of the nearest facility in self.__X.
        """

        self.__stime = (0.0, 0.0)
        """Start time of a solution procedure."""
        
        self.__ctime = (0.0, 0.0)
        """Computation time (CPU and Walltime) required to solve a problem instance.""" 
        
        self.__mip_time = 0.0
        """Computation time (walltime seconds) spent by the MIP solver."""
        
        self.__cg_iter = float('inf')
        """Maximal number of column generation iterations."""
        
        self.__pricing_method = 'Drezner'
        
        self.__timLimit = None 
        """Time limit to be applied when solving a problem as a MIQCP using 
        a MIP solver."""
        
        self.__strategy = 0 
        """Strategy to be used by the MIQCP solver: If 0, the solver decides 
        automatically; if 1 the quadratic continuous relaxation is used 
        for obtaining lower bounds; if 2 conical cuts ar used. See also
        docplex.mp.model.parameters.mip.strategy.miqcpstrat and 
        gurobipy.model.params.MIQCPMethod."""
        
        self.__nodes_processed = 0
        """Number of nodes processed in total for solving a problem with the MIP solver."""

        if not fname is None:
            Y, w, names = read_points(fname) 
            g_coo = return_lola()
            assert not Y is None, "Unable to read file"+fname
        if not Y is None:
            self.set_data(Y, geospatial=geospatial, w=w, g_coo = g_coo, names=names)
        
    #---------------------------------------------

    def set_data(self, Y, w=None, geospatial=False, g_coo=None, names=None):
        """
        Pass data to this instance of a planar location problem.

        Parameters
        ----------
        Y : mx2 numpy array of float
            Coordinates of the m customer points.
        w : numpy array of int or float or None
            Array of customer weights. If None all weights will
            be set to 1.
        geospatial : bool, optional
            If True, the coordinates in Y are assumed to be longitude
            and latitude. Default: False.
        g_coo : tuple of two lists, optional
            If not None and Y does not already contain geospatial coordinates,
            g_coo should a tuple of two lists, the first one giving the
            longitudes, the second one the latitudes of the customer points.
        names : list of str or None
            Addresses/names of the m customer points. 
        """ 
        self.__names = names
        if geospatial: 
            self.__lo = Y[:, 0]
            self.__la = Y[:, 1]
            origin = (min(self.__lo), min(self.__la))
            self.__Y = np.array(list(map(lambda y: lla2xy(y[0], y[1], origin), Y)))
        else:
            self.customers = Y 
            if not g_coo is None: 
                self.__lo = g_coo[0]
                self.__la = g_coo[1]
        self.weights = w   
        self.__sum_dist = None 
        self.__sum_wdist = None 
        self.__max_dist = None 
        self.__max_wdist = None 
        self.__assigned = None 
        self.__X = None 
            
    #---------------------------------------------
    
    def __get_dist(self):
        """
        Compute total weighted distance, total distance, maximal weighted 
        distance and maximal distance to nearest facility location.
        """
        p = 1 if len(self.__X.shape)==1 else self.__X.shape[0]
        m = self.__Y.shape[0]
        W = self.__w[0]
        weighted = np.any( self.__w != W )
        if p==1:
            dists = np.fromiter( (euclid(self.__X,y) for y in self.__Y), dtype=float )
        else: 
            a = self.__assigned 
            if a is None: 
                a = [np.argmin(np.fromiter((euclid(self.__X[j], y) for j in range(p)), float)) for y in self.__Y]
                self.__assigned = a
            dists = np.fromiter( (euclid(self.__X[a[i]], self.__Y[i]) for i in range(m)), dtype=float)
        self.__sum_dist = dists.sum()
        self.__max_dist = dists.max()
        self.__sum_wdist = np.dot(self.__w,dists) if weighted else self.__sum_dist*W 
        self.__max_wdist = (self.__w*dists).max() if weighted else self.__max_dist*W
     
    #---------------------------------------------
    
    @property 
    def customers(self):
        """The customer point coordinates."""
        return self.__Y 
    
    @customers.setter 
    def customers(self, Y):
        """Sets the customer coordinates to the numpy array Y.
        Note that Y is an array of 2-dimensional arrays."""
        if isinstance(Y,np.ndarray) and Y.shape[1]==2:
            self.__Y = Y 
        else:
            print("Coordinates must be a numpy array of 2-dimensional arrays!")
    
    @property
    def weights(self):
        """The customer point weights."""
        if self.__w is None: return np.ones(len(self.__Y), dtype=int) 
        return self.__w
    
    @weights.setter 
    def weights(self, w):
        """Set the customer point weights to the array w. Note
        that customer weights need to be positive."""
        if w is None and not self.__Y is None:
            self.__w = np.ones(len(self.__Y), dtype=int )
        elif isinstance(w,np.ndarray) and np.all(w > 0):
            self.__w = w
        else: 
            print("Weights must be a positive numpy array!")

    @property 
    def names(self):
        """Addresses/names of the customer points."""
        return self.__names 

    @property 
    def lola(self):
        """Longitude and latitude coordinates of the customer points"""
        return self.__lo, self.__la 

    @property 
    def weighted_distance(self):
        """Sum of weighted distances to nearest facility in a solution."""
        return self.__sum_wdist 
    
    @weighted_distance.getter 
    def weighted_distance(self):
        """Sum of weighted distances to nearest facility in a solution."""
        if not self.__sum_wdist is None: return self.__sum_wdist
        if not self.__X is None: 
            self.__get_dist()
            return self.__sum_wdist 
        return 0
    
    @property 
    def distance(self):
        """Sum of (unweighted) distances to nearest facility in a solution."""
        return self.__sum_dist 
    
    @distance.getter 
    def distance(self):
        """Sum of (unweighted distances to nearest facility in a solution."""
        if not self.__sum_dist is None: return self.__sum_dist
        if not self.__X is None: 
            self.__get_dist()
            return self.__sum_dist
        return 0
    
    @property 
    def max_weighted_distance(self):
        """The maximal weighted distance to nearest facility in a solution."""
        return self.__max_wdist 
    
    @max_weighted_distance.getter 
    def max_weighted_distance(self):
        """Returns the maximal weighted distance to nearest facility in a solution."""
        if not self.__max_wdist is None: return self.__max_wdist
        if not self.__X is None: 
            self.__get_maxdist()
            return self.__max_wdist
        return 0
    
    @property 
    def max_distance(self):
        """The maximal (unweighted) distances to nearest facility in a solution."""
        return self.__max_dist 
    
    @max_distance.getter 
    def max_distance(self):
        """The maximal (unweighted) distance to nearest facility in a solution."""
        if not self.__max_dist is None: return self.__max_dist
        if not self.__X is None: 
            self.__get_maxdist()
            return self.__max_dist
        return 0
    
    @property
    def lowBound(self):
        """Lower bound on objective value (only for multi-source Weber if 
           solved by column generation)."""
        return self.__lobnd    
    
    @property
    def facilities(self):
        """Array of the p facility coordinates."""
        return self.__X 
    
    @property
    def assigned(self):
        """Assignment of customer locations to facility (indices)."""
        if len(self.__X.shape)==1 or self.__X.shape[0] == 1: return [0]*len(self.__Y)
        return self.__assigned
    
    @property 
    def mip_time(self):
        """The MIP solver's computation time (wall time) in seconds."""
        return self.__mip_time 
    
    @property 
    def mip_nodes(self):
        """The number of branch-and-cut tree nodes enumerated."""
        return self.__nodes_processed 
    
    @property 
    def cpuTime(self): 
        """Returns CPU time in seconds"""
        return self.__ctime[0]

    @property 
    def wallTime(self): 
        """Computation time (walltime) in seconds"""
        return self.__ctime[1]
    
    @property 
    def mipSolver(self):
        """
        The MIP solver used for solving MIPs.
        """
        return set_mipSolver( )
    
    @mipSolver.setter 
    def mipSolver(self, solver ):
        set_mipSolver(solver)
        
    @property 
    def timeLimit(self):
        """
        The time limit to be applied for the MIQCP solver.
        """
        return self.__timLimit 
    
    @timeLimit.setter
    def timeLimit(self, value):
        """
        Set the time limit to be applied for the MIQCP solver.
        """
        self.__timLimit = value 
        
    @property 
    def miqcp_strategy(self):
        """
        The MIP solver's MIQCP solving strategy.
        """
        return self.__strategy 
    
    @miqcp_strategy.setter
    def miqcp_strategy(self, value):
        """
        Set the strategy (0, 1, or 2) to be applied for the 
        MIQCP solver.
        """
        if value in (0,1,2): self.__strategy = value 
        
    @property 
    def cg_iter(self):
        """
        The maximal number of column generation iterations
        to be done within the column generation method for
        the multi-source Weber problem. 
        """
        return self.__cg_iter 
    
    @cg_iter.setter 
    def cg_iter(self, value):
        """
        Set the maximal number of column generation iterations
        to the given value.
        """
        if value > 0: self.__cg_iter = value
        
    @property 
    def pricing_method(self):
        """
        The method to solve the pricing problem within
        a column generation for the multi-source Weber problem.
        """
        return self.__pricing_method 
    
    @pricing_method.setter
    def pricing_method(self, value):
        """
        Use method *value* for solving the pricing problem within
        a column generation for the multi-source Weber problem.
        value must equal *Drezner* or *MIQCP*
        """
        self.__pricing_method = value  
        
    @property 
    def optTol(self):
        """
        The relative optimality tolerance value to 
        be used within the column generation.
        """
        return set_optTol()
    
    @optTol.setter 
    def optTol(self, value):
        """
        Set the column generation's relative optimality
        tolerance to the given value.
        """
        set_optTol(value)
    
    #---------------------------------------------
    
    def solve(self, p=1, minisum=True, method=None, silent=False):
        """
        Solves an instance of a planar location problem.
        
        Parameters
        ----------
        p : int 
            Number of facility locations to find (default=1).
        minisum : bool
            If True, a minisum location problem (Fermat-Weber,
            or multi-source Weber) is solved; otherwise the
            1-center (p=1) or p-center problem is solved.
            (default=True).
        method : str 
            Determines the method to be applied for solving
            the problem. Note that *method* need to be specified
            except if Drezner's method for solving the 
            Fermat-Weber problem should be applied. The following
            methods are available:
            
            1. For the Fermat-Weber problem (p=1, minisum=True)
            
            - *Drezner* : Drezner's method (Default method).
            - *Ostresh-val* : Ostresh's method with step size parameter lambda 
              set to the value *val*. For example, Ostresh-2. If *val* is not 
              specified, the step size parameter is set to 2.
            - *Weiszfeld* : Weiszfeld's method (same as Ostresh-1).
            - *SOCP* : Solves the problem using the MIP solver for 2nd order cone problems.
            
            2. For the 1-center problem in plane 
            
            - *PrimDual* : A primal-dual convex optimization method.
            - *Elzinga* : Elzinga and Hearn's method (can only be 
              applied if all weights are identical).
            - *Charalambous* : Charalambous' method.
            - *Welzl* : Welzl' method.
            - *SOCP* : Solves as 2nd order cone problem using the MIP solver.
            
            The default is *Elzinga* for equal weights and *Charalambous*
            otherwise.            
                
            3. For the multi-source Weber and planar p-center problem:
            
            - *LOCA-num* : Location allocation heuristic. If *num* is specified, 
              then *num* is the number of times the procedure is repeated with 
              different random initial solutions. If *num* is not specified, 
              clustering methods are used to obtain initial solutions.
            - *pmedian* : p-median heuristic for the multi-source Weber problem.
            - *VNS* : A variable neighbourhood search heuristic, that is,
              a local search procedures that tries to improve tjhe 
              current solution. If no initial solution has already 
              been obtained, LOCA is used to this end.   
            - *ColGen* : Column generation procedure.
            - *MIPQ* : Applies the MIP solver for solving the problem when modelled as 
              a 2nd order cone MIP. 
            - *Ostresh* : Ostresh's method for the two-Weber problem.
                           
            If no method is specified, the default is applied, which is LOCA 
            if p > 2 and Ostresh for p=2.
        """
        X, a = None, None  
        screen = 'off' if silent else 'on'
        self.__stime = (time.process_time(),time.time())
        if p==1:
            # Single facility location problem
            if minisum:
                # Solve the Fermat-Weber problem 
                meth = 'drezner' if method is None else method.lower()
                meth = meth.split('-')
                l = 1 if len(meth)==1 else float(meth[1])
                if 'weis' in meth[0]: meth[0]='weiszfeld'
                if 'ostr' in meth[0]: 
                    meth[0]='ostresh'
                    if l==1: l=2
                if 'socp' in meth[0]:
                    _, X = SOCPweber( self.__Y, self.__w, screen )
                else:
                    _, X = solveWeber( self.__Y, self.__w, l, meth[0], screen )
            else:
                # Solve the center problem in plane  
                if method is None:
                    meth = 'chara' if np.any(self.__w != self.__w[0]) else 'elzinga'
                else:
                    meth = method.lower()
                if 'elzin' in meth:
                    _, X = elzinga_hearn(self.__Y, screen)
                elif 'chara' in meth:
                    _, X = charalambous( self.__Y, self.__w, screen )
                elif 'welzl' in meth: 
                    _, X = welzl( self.__Y, self.__w, screen)
                elif 'prim' in meth:
                    _, X = growRadius( self.__Y, self.__w, screen )
                elif 'soc' in meth: 
                    _, X = SOCPcenter( self.__Y, self.__w, screen )
        else:
            # Solve multi-facility, that is, a location-allocation problem
            default_meth = 'loca' if p > 2 else 'Ostresh'
            meth = default_meth if method is None else method
            meth = meth.split('-')
            num = 0 if len(meth)==1 else int(meth[1])
            meth[0] = meth[0].lower()
            if 'loca' in meth[0]:
                if num==0: 
                    _, X, a, _ = locAlloc( p, self.__Y, self.__w, minisum=minisum,\
                                           initLA='cluster', screen=screen )
                else:
                    _, X, a, _ = locAlloc( p, self.__Y, self.__w, minisum=minisum,\
                                           repeat=num, screen=screen)
            elif 'pmed' in meth[0]:
                _, X, a, _ = pmedian_heuristic( p, self.__Y, self.__w, minisum, screen )
            elif 'vns' in meth[0]:
                p_old = 1 if (self.__X is None or len(self.__X.shape)==1) else self.__X.shape[0] 
                X0 = None if p != p_old else self.__X 
                _, X, a = weber_vns( p, self.__Y, self.__w, X0, minisum, screen ) 
            elif 'mip' in meth[0]:
                if minisum:
                    _, X, a, tim, nnodes = weber_mipq( p, self.__Y, self.__w, screen,\
                                                       self.__strategy, self.__timLimit)
                else: 
                    _, X, a, tim, nnodes = pcenter_mipq( p, self.__Y, self.__w, screen,\
                                                         self.__strategy, self.__timLimit)
                self.__mip_time = tim 
                self.__nodes_processed = nnodes
            elif 'col' in meth[0] and minisum:
                #_, LB, X, a, _ = colgen( p, self.__Y, self.__w, mip_pricing=False, screen=screen)
                _, LB, X, a, _ = colgen( p, self.__Y, self.__w, screen=screen,\
                                         mip_pricing = 'MI' in self.__pricing_method,\
                                         max_iter=self.__cg_iter )
                self.__lobnd = LB 
            elif p==2:
                _, X, a = twoFacility( self.__Y, self.__w, minisum, screen )   
        
        if not X is None:            
            self.__ctime = (time.process_time()-self.__stime[0],time.time()-self.__stime[1])  
            self.__X = X       
            if not a is None: self.__assigned = a 
            self.__get_dist()
            
    #---------------------------------------------
    
    def plot(self):
        """
        Plots a solution.
        """
        plot_points( self.__Y, self.__X, lola = (self.__lo,self.__la), cust_id=self.__names )        
