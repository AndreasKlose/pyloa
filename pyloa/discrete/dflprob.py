"""
    Module discrete.dflprob of package pyloa:
    Base class for a discrete facility location problem.
"""
import math
import numpy as np
from os.path import basename
from timeit import time
from pyloa.util import int_array, real_array
from pyloa.mip.model import set_mipSolver
from pyloa.discrete.parser import read_cflp, read_uflp, read_tscflp, read_tlcflp

#--------------------------------------------------------------------------

class DFLProblem:
    """
    Base class for a discrete facility location problem (single and two stage).
    """
    
    def __init__( self, fname=None, formt='CFL-GK', unitCost=True, scale=1, mcf=False, \
                  d=None, s=None, f=None, c=None ):
        """
        Creates an instance of a discrete facility location problem.
        
        Parameters
        ----------   
        fname : str, optional  
            None or name of the input file or problem instance to be solved.
            If None, it is assumed that the data are either provided via arguments
            c, f, d and s or later passed to the class instance using the 
            method "set_data".  
        formt : str, optional  
            Format of the input file, that is, 
               * 'CFL-AO'     for the old Avella-Boccia CFLP instances  
               * 'CFL-AN'     for the new Avella-Boccia CFLP instances  
               * 'CFL-G'      for the Guastaroba CFLP instances  
               * 'CFL-GK'     for the Goertz-Klose CFLP instances (the default)  
               * 'CFL-OR'     for an CFLP instance in OR library format  
               * 'UFL-OR'     for an UFLP instance in CFLP OR library format similar  
                              to the M-instances from UflLib (Max Planck Informatik)  
               * 'UFL-SIMPLE' for the UflLib instances in the "simple" format 
               * 'TSCFL'      for the 2-stage instances from Klose (1999,2000)
               * 'TLCFL-AK'   for 2-level instances using the similar file format
                              as the 2-stage instances from Klose (1999,2000)
               * 'TLCFL-FRA'  for 2-level instances from Fernandes et al (2014).
        unitCost : bool, optional   
            If True, the supply cost c[j][i] (resp. c[i,j,k]) are expected to 
            be the unit cost of supplying customer i from facility j (resp.
            to supply k via facilities i and j). Otherwise, it need to be 
            the cost of supplying all of customer i's demand from the facility
            (or facilities in case of a two-level problem). If data are read 
            from file, unitCost is set to false in case of formt is from
            ('GK','OR','SIMPLE','TSCFLP','TLCFLP'). Note that unitCost requires 
            that demand data are passed to this class instance. In case that d 
            remains None, unitCost will be set to false and the problem assumed
            to be uncapacitated. For a two-level problem (location decisions 
            on both levels), the cost matrix can also be a 3d-array. Then unitcost=True 
            indicates if c[i,j,k] is the unit cost or the cost to meet all of customer 
            k's demand from facility pair (i,j). (Default=True)  
        scale : int or float, optional  
            Is only used for Guastaroba instances. In case of these instances,
            demands and capacities might be non-integral. They are, therefore,
            multiplied by scale and then rounded to integers (demands rounded
            down, capacities to the nearest integer). Unit cost data are
            accordingly divided by scale.
        mcf : bool, optional  
            Is only used in case of a two-level problem where fname is defined
            and the data thus read from file. If mcf is True, then the 
            multi-commodity formulation of the problem will be used and the
            cost matrix be a 3-dimensional so that c[i,j,k] is the (unit)
            cost to supply customer k from facilities i and j. Note that
            if argument c is defined (and fname not), then the multi-commodity
            formulation will be used if c is a 3d-numpy array.
        d : numpy array of int, optional   
            Customer demands. d can be None (see argument c for exceptions). 
            Then either fname should be defined or the data passed later to an 
            instance of this class using the method set_data. 
        s : Two or a single numpy array of int, optional  
            If s is a single array, it defines the facility capacities of an
            ordinary CFLP. If s is None, the capacities are either provided by 
            an input file (cf. fname) or unlimited capacities are assumed. 
            Alternatively, the data can also later be passed to an instance of this 
            class using the method set_data. If s is two arrays, then it should be 
            a tuple of two arrays such that s[0] gives the array of plant capacities 
            and s[1] the array of depot capacities. Unlimited capacities on level i 
            are assumed if s[i] is None. If both are None or s itself is None, the 
            data either need to come from an input file (cf. fname) or unlimited 
            capacities are assumed. 
        f : Two or a single numpy array of float, optional
            If f is a single array, f is assumed to be the fixed facility cost
            in an ordinary CFLP or UFLP or the depot fixed costs in a two-stage
            problem where there are no location decisions on the plant level 
            (stage 1). Otherwise, for a two-level problem, f is a tuple two arrays 
            such that f[0] is the plant fixed cost array and f[1] the depot fixed 
            cost array. If f is None, the data may come from an input file (cf. argument 
            fname) or be passed later to the class instance using the method set_data.
        c : A single numpy 2-array or two numpy 2d-arrays or 3d-numpy array of float, optional  
            In case of an ordinary CFLP or UFLP, c[j][i] is either the unit or the 
            total cost of supplying customer i from facility j. For a two-stage
            problem (no location decisions on the first stage), c must be a 
            tuple of two numpy 2d-arrays such that c[0][i,j] is the cost
            per unit (!) of product to supply facility j at stage 2 (depot) from
            a facility i at stage 1 (plant) and c[1][j,k] is the cost per unit
            (unitCost=True) or the cost to supply all of customer k's demand
            (unitCost=False) from facility j (depot) at stage 2. In case of a 
            two-level problem (location decisions on both stages), c can either be 
            a tuple of two matrices just as in the two-stage case or be a numpy 3d-array.
            In the latter case, c[i,j,k] is either the cost per unit (unitCost=True)
            or the cost to supply all of customer k's demand (unitCost=False) from 
            a facility pair i,j (where i is the facility, 'plant', on the first 
            level and j the facility, 'depot', on the second level). 
            Note that it will be assumed that the problem is a two-level problem 
            if c is a 3d-array (otherwise, it will be assumed to be of that type if 
            f is a tuple consisting of two arrays). If c is a 3d-array, the
            two-level problem will (and need to) be modelled using three-index
            variables for the flow variables (multi-commodity formulation). Note 
            that c can be None. In this case, the data might come from a file 
            (cf. argument fname) or are passed later to an instance of this class 
            using the class method set_data.
            Note also that if c is a tuple of two-matrices, the first one is unit cost
            of distribution from the first to the second stage (plants->depots).
            In that case, d need to be defined and cannot be None, as otherwise
            cost for forwarding the customers demand on the first stage of
            distribution cannot be determined! 
            
        References
        ----------
        Fernandes, D.R.M., Rocha, C., Aloise, D., Ribeiro, G.M., Santos, E.M, Silva, A.
        (2014). A simple and effective genetic algorithm for the two-stage capacitated
        facility location problem. Computers and Industrial Engineering 75:200-208.
        
        Klose, A. (1999). An LP-based heuristic for two-stage capacitated facility location 
        problems. Journal of the Operational Research Society 50:157-166.
        
        Klose, A. (2000). A Lagrangean relax-and-cut approach for the two-stage capacitated 
        facility location problem. Eur. J. Oper. Res. 126:185-198.

        """
        # Check if data come from file defining a CFLP, TSCFLP or TLCFLP
        is_tscflp =  'TSCFL' in formt.upper() 
        is_tlcflp = 'TLCFL' in formt.upper() # Data file contains two-level CFLP
        is_cflp = not (is_tscflp or is_tlcflp) and 'CFL' in formt.upper()
        
        self.__stages = 2 if is_tscflp or is_tlcflp else 1
        """Number of distribution stages (equals 1 for the ordinary CFLP and UFLP)."""
        
        self.__two_level = is_tlcflp
        """False if there are location decisions only on one distribution stage
        as in case of the ordinary UFLP and CFLP or in case of the TSCFLP."""
                 
        self.prob_name = None if fname is None else basename(fname).split('.')[0]
        
        self.__c = c 
        """c[j,i] is the cost to supply either one unit or all of customer i's demand
        from a facility at site j. In case of a two-stage problem, c[1][i,j] is
        this cost and c[0][i,j] will be the cost of sending one unit of product
        from a plant at site i to a depot/facility at site j. For a two-level
        problem, c can also be a 3d-array."""
        
        self._c_is_scaled = False 
        """Will be set to True if the supply cost matrix is scaled so that 
           min_j c[j,i] = 0 for each customer i. If c is 3d, then min_{ij} c[i,j,k]
           will be zero in this case for each customer k."""
           
        self._cmin = None 
        """When supply costs are scaled, this will keep min_j c[j,i] for each 
        customer i (min_{ij} c[i,j,k] in the 3d-case for each customer k)."""
        
        self.__s = s
        """In case of the classical CFLP, s[j] is the capacity of a facility at site j.
        For a two-stage problem, s[1][j] gives this figure and s[0][k] will be the
        capacity of plant k. It is allowed to have s=None (resp. s[0]=None or s[1]=None),
        as this will be interpreted as unlimited capacity."""
        
        self.__uncap = False if self.stages==1 else (False,False)
        """True if the instance is actually uncapacitated, that is a classical UFLP
        in case of a one-stage problem. Otherwise, for a two-stage or two-level problem,
        _uncap is a tuple and _uncap[i] will be false if there are no capacity limits 
        on the i-th stage"""
        
        self.__d = d 
        """Customer demands"""
         
        self.__totD = 0
        """Total customer demand"""
        
        self.__f = f 
        """Fixed facility/depot costs for a one-stage problem and the two arrays 
        of fixed plant and depot cost in case of a two-level problem with location
        decisions on both stages."""
             
        self.facilities = None  
        """List of open facilities in a solution (one-stage problem or two-stage
        with fixed plant sites) or a tuple of lists of open plant and depot sites 
        for a two-level problem.""" 
           
        self.assigned = None 
        """In case of an uncapacitated one-stage problem, the assignment of customers to 
        facilities."""
        
        self.supply = None
        """For an ordinary CFLP (limited capacities), it is a dictionary of the positive
        flows from open facilities (depots) to customers, that is, supply[(j,i)] is the 
        percentage of customer i's demand supplied from a facility with index j. If
        the CFLP actually is instance of an UFLP, the allocation of demands to 
        facilities follows from the attribute self.assigned.
        For a two-stage problem or a two-level problem modelled with 2-index flow
        variables, supply is a tuple of two dictionaries, where the first one gives
        the amount of flow from plants to depots and the second the flow from
        depots to customers expressed as fractions of a customer's demand.
        For a two-level problem modelled using the multi-commodity formulation,
        supply is again a single dictionary showing the positive flows as percentage
        of demands from plants via depots to customers."""
           
        self.fcost = 0.0 
        """Total fixed cost (depots) of a solution for a one-level problem or a
        two-stage problem (fixed plant locations). For a two-level problem,
        fcost[0] and fcost[1] are the total plant and depot fixed cost, resp."""  
        
        self.tcost = 0.0 
        """Total supply/transportation cost (depots->customers) of a solution
        for a one-stage problem or for a two-level problem modelled using the
        multi-commodity flow formulation. Otherwise, that is for a two-stage
        or a two-level problem with 2-index flow variables, tcost[0] and tcost[1] 
        are the plant-depot and depot-customer transportation costs, resp."""
        
        self.cost = 0
        """Total cost of a solution"""
        
        self.ccnst = 0 
        """Constant in the total cost to supply the customers"""
        
        self._c_scale = 1
        """ Factor by which cost data are divided to avoid too large numbers""" 
        
        self.bound = 0 
        """Best objective value lower bound obtained""" 
        
        self.__unitCost = unitCost
        """If True, if costs[j,i] (costs[1][j,i] or costs[i,j,k] for a two-stage or
           two-level problem) are assumed to be unit cost; otherwise total cost 
           of supplying customer i's  demand from a facility/depot at site j."""  
        
        self.__model = None 
        """Instance of the MIP solver model"""
        
        self.__x = None 
        """Demand allocation decision variables"""
        
        self.__y = None 
        """Locational decision variables (depots)"""

        self.__z = None 
        """
        Locational decision variables for the first stage (Plants)
        """

        self.__v = None 
        """
        Decision variables for the flow from plants to depots.
        """

        self.__silent = False 
        """If False, solvers send log-output to stdout"""
        
        self.__stime = (0.0,0.0)
        """Start time of a solution procedure"""
        
        self.ctime = (0.0,0.0)
        """Computation time (CPU and Walltime) required to solve a problem instance""" 
        
        self.mip_time = 0.0
        """Computation time (walltime seconds) spent by the MIP solver"""
        
        self.mip_work = 0
        """Deterministic computation time. For Cplex this is the total amout
        of CPU ticks. For GuRoBi one work unit is about one second on single
        thread."""
        
        self.nodeCount = 0
        """Number of nodes processed in total by branch-and-bound / branch-and-cut"""
        
        self.__useBenders = 0
        """Possible values are 0, 1, and 2. If 0, no Benders' decomposition is
        applied. If 1 and Cplex is used as MIP solver, Cplex's built-in 
        Benders' decomposition is used. If 2, a dedicated Benders' decomposition
        is performed. Note that the latter is not yet implemented."""
         
        self.__useLB = False 
        """Applies only if Cplex is used to solve the problem's MIP formulation.
        If True, Cplex built-in local branching heuristic will be activated,
        otherwise not (the default)."""
        
        self.__addVubs = 2
        """Indicates how to treat variable upper bound constraints. If 0, they are not 
           included to the initial model; if 1 they are included as user cut constraints 
           (but not via a callback) if Cplex is used as MIP solver and via a callback
           if GuRoBi is the MIP solver. If 2, variable upper bounds are included 
           explicitly in the initial model.""" 
           
        self.__cuts = -1 
        """Cutting strategy to be used by the MIP solver. If -1, the MIP solver
        decides, if 0 cuts are switched off, if 2 cuts are generated aggressively,
        if 3 cuts are generated very aggressively. If Cplex is the MIP solver,
        the latter even sets disjunctive cuts to aggressive mode."""
           
        self.__mipStrategy = 0
        """Only applies if Cplex is used to solve the problem's MIP formulation.
        This is Cplex's dynamic search switch. A value of 1 means that Cplex
        uses traditional branch-and-cut, 2 means it uses dynamic search, 0
        means Cplex decides automatically (which is almost always dynamic search)."""
        
        self.__timeLim = 0
        """Time limit for the solver"""
        
        self.__nodeLim = -1
        """Limit on number of nodes in enumeration tree"""
        
        self.__optTol = 1.0E-07 
        """Optimality tolerance to be used for a MIP solver."""

        if fname is None:
            if not (f is None or c is None):
                self.set_data( f, c, d, s, unitCost=unitCost )
        else:
            # Read data from file fname 
            formt = formt.split('-')[1].upper() if '-' in formt else formt
            unitCost = not formt in ['AK','GK','OR','SIMPLE','TSCFL'] 
            try:
                assert formt in ['AK','AO','AN','G','GK','OR','SIMPLE','TSCFL','FRA']
            except: 
                print('File format parameter formt need to be CFL-AO, CFL-AN, CFL-G',end=', ')
                print('CFL-GK, UFL-OR, UFL-SIMPLE, TSCFL, TLCFL-AK, or TLCFL-FRA.')
            if is_cflp:
                d, s, f, c = read_cflp( fname, formt=formt, scale=scale )
                self.set_data( f, c, d, s, unitCost=unitCost )
            elif is_tscflp: 
                d, s, f, c = read_tscflp( fname )
                self.set_data( f, c, d, s, unitCost=unitCost )
            elif is_tlcflp:
                d, s, f, c = read_tlcflp(fname,mcf=mcf) if formt=='FRA' else \
                             read_tscflp(fname, two_level=True, mcf=mcf) 
                self.set_data( f, c, d, s, unitCost=unitCost )
            else:
                f, c = read_uflp( fname, formt=formt )
                self.set_data( f, c )
                                                               
    #---------------------------------------------
    
    def __check_if_uncap(self):
        """
        Checks if it is an uncapacitated problem 
        """
        if self.__stages==1:                   
            self.__uncap = self.__s is None or self.__d is None or self.__s.min() >= self.__totD
            if not self.__s is None: self.__s[:] = np.minimum( self.__s, self.__totD )
        else: 
            ucap = [True, True] 
            for i in (0,1): 
                if not self.__s[i] is None:
                    ucap[i] = self.__s[i].min() >= self.__totD
                    self.__s[i][:] = np.minimum( self.__s[i], self.__totD )
            self.__uncap = (ucap[0], ucap[1])    
    
    #---------------------------------------------
    
    def set_data(self, f, c, d=None, s=None, unitCost=True ):
        """
        Pass data to this instance of a facility location problem.
        
        Parameters
        ----------
        f : One or two arrays of float
            Fixed facility costs in an ordinary UFLP or CFLP as well as
            the depot fixed cost for a two-stage problem (no location decision
            on the first level). For a two-level problem, f[0] and f[1] are
            the fixed facility cost arrays on each stage/level.
        c : A single or two numpy 2d-array or a 3d-numpy array of float
            In case of an ordinary CFLP or UFLP, c[j][i] is either the unit or the 
            total cost of supplying customer i from facility j. 
            For a two-stage problem, c must be a tuple of two 2d-numpy arrays 
            so that c[1][j,i] is the same cost matrix as in the one stage case 
            and c[0][i,j] is the cost of shipping one unit of product from plant i 
            (stage 0) to a depot (facility at stage 1) at site j. 
            If c is a 3d-array, the problem is assumed to be a two-level problem to
            and to be modelled using the multi-commodity flow formulation, that is 
            c[i,j,k] is the cost to meet either one unit or all of customer k's demand 
            from the pair (i,j) of facilities in case of a two-level problem where i is the 
            facility at level 1 and j on level 2. For a two-level problem that
            should not be modelled using 3-index flow variables, c need to be
            a tuple of two 2d-matrices just as in the case of a two-stage problem.
            Note that if c is two-matrices, the first one is always unit cost. That
            also means that in this case d cannot be None, as demand data need to
            be known to compute costs on the first stage of distribution.   
        d : numpu array of int, optional
            Customer demands. d can be None in case of an uncapacitated
            problem.
        s : One or two numpy arrays of int, optional
            Facility capacities in an ordinary UFLP or CFLP. For a two-stage
            or two-level problem, s[0] and s[1] are the arrays of facility
            capacities on each stage/level.
        unitCost : bool, optional 
            If true, c[j][i] (c[1][j,i] or also c[i,j,k] in case of two stages/levels) 
            is expected to be the unit cost of supplying customer i from facility j. 
            Otherwise, it need to be the cost of supplying all of customer i's demand from 
            facility j. Note that unitCost requires that demand data are passed to this 
            class instance. In case that d remains None, unitCost will be set to false
            and unlimited capacities assumed. (Default=True) 
        """
        self.__stages = 1 if isinstance(c,np.ndarray) and c.ndim==2 else 2
        self.__two_level = self.__stages > 1 and isinstance(f,tuple)
        
        self.__d, self.__s, self.__f, self.__c = d, s, f, c
        self.__unitCost = False if d is None else unitCost
        self.__totD = 0 if d is None else d.sum()
        
        self.__check_if_uncap()
                                   
    #---------------------------------------------
    
    def _set_starttime( self ):
        """
        Initialize start time of a solving process.
        If cpx_time is True, docplex is told to 
        measure in CPU time.
        """
        self.__stime = (time.process_time(),time.time())
        
    #---------------------------------------------
 
    def _set_comptime( self, mip_time=False ):
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
    
    def scale_costs(self):
        """
        Rescale the supply cost such that the smallest supply cost for 
        each customer equal zero. Note that this just adds a constant 
        to the objective value. The constant amounts to the sum of the 
        unscaled cheapest cost.
        """
        c = self.__c if isinstance(self.__c, np.ndarray) else self.__c[1]
        axe = 0 if c.ndim == 2 else (0,1)
        cmin = c.min( axis=axe )
        cmax = c.max( axis=axe )*self.__d if self.__unitCost else c.max(axis=axe)  
        c -= cmin       
        self.ccnst = np.dot(cmin,self.__d) if self.__unitCost else cmin.sum()
        self._cmin = cmin 
        
        # Reduce supply and fixed cost if they appear to be very large
        fsum = self.__f[0].sum() + self.__f[1].sum() if self.__two_level else self.__f.sum()
        csum = fsum + cmax.sum() 
        cimax = cmax.max()
        big = 9999999.0
        k1 = math.ceil( math.log(csum,10)-math.log(big,10) ) if csum > big else 0
        k2 = math.ceil( math.log(cimax,10)-6 ) if cimax > 1.0E+06 else 0
        c_scale = 10**max(k1,k2)
        if c_scale > 1:
            if self.__stages == 1:
                self.__f /= c_scale 
                self.__c /= c_scale
            else:
                if self.two_level: 
                    for F in self.__f: F /= c_scale
                else: 
                    self.__f /= c_scale
                if isinstance(self.__c, np.ndarray):
                    self.__c /= c_scale 
                else: 
                    for C in self.__c: C /= c_scale
                        
        self._c_scale = c_scale 
        self._c_is_scaled = True
         
    #---------------------------------------------
    
    def unscale_costs(self):
        """
        Undo the scaling of supply costs.
        """
        if self.__stages == 1:
            if self._c_scale > 1:
                self.__f *= self._c_scale 
                self.__c *= self._c_scale
            self.__c += self._cmin  
        else:    
            if self._c_scale > 1:
                if self.__two_level:
                    for F in self.__f: F*self._c_scale
                else:
                    self.__f *= self._c_scale 
            if isinstance(self.__c,np.ndarray):
                if self._c_scale > 1: self.__c *= self._c_scale
                self.__c += self._cmin 
            else: 
                if self._c_scale > 1:
                    for C in self.__c: C *= self._c_scale 
                C = self.__c[1]
                C += self._cmin 
        self.ccnst = 0.0
        self._cmin = None
        self._c_is_scaled = False 
            
    #---------------------------------------------
    
    @property
    def c(self):
        """
        Returns the supply cost matrix/matrices.
        """
        return self.__c 
    
    @c.setter 
    def c( self, cst_mat ):
        """Set the supply cost matrix/matrices to the supplied data."""
        if isinstance( cst_mat, np.ndarray ) and cst_mat.ndim >= 2:
            self.__c = cst_mat
            self.__stages = 1 if cst_mat.ndim==2 else 2
        elif isinstance( cst_mat, tuple ) and isinstance( cst_mat[0], np.ndarray )\
             and isinstance( cst_mat[1], np.ndarray ) and cst_mat[0].ndim==2 \
             and cst_mat[1].ndim==2:
            self.__c = cst_mat
            self.__stages = 2
            self.__two_level = False # If also fixed cost is a tuple then its 2-level
            
    @property
    def d(self):
        """Returns the array of customer demand."""
        return self.__d 
    
    @d.setter 
    def d( self, value ):
        """
        Sets array of customer demands to the given value.
        See pyloa.util.int_array for more information. 
        Note that demand values need to be non-negative integers.
        """
        m, c = 0, self.__c
        if not self.__d is None: 
            m = len(self.__d)
        elif not c is None:
            m = c.shape[2] if c.ndim==3 else c.shape[1] 
        self.__d = int_array( value, dim=m, down=True )
        self.__totD = 0 if self.__d is None else self.__d.sum()
        self.__check_if_uncap()
        
    @property 
    def totalDemand(self):
        """Return the total customer demand"""
        return self.__totD
        
    @property 
    def s(self):
        """Returns the array(s) of facility capacities"""
        return self.__s 
    
    @s.setter  
    def s( self, value ):
        """
        Sets the array(s) of facility capacities to the
        given values. See pyloa.util.int_array for the
        result. In case of a two-stage or two-level problem,
        value must be a tuple specifying the capacities
        for both stages. 
        """
        if not self.__s is None: 
            if isinstance(self.__s,tuple):
                p, n = len(self.__s[0]), len(self.__s[1])
            else:
                n = len(self.__s) 
        elif not self.__f is None:
            if isinstance( self.__f, tuple ):
                p, n = len(self.__f[0]), len(self.__f[1]) 
            else:
                n = len(self.__f) 
        else:
            p, n = 0, 0
                
        if isinstance(value, tuple) and len(value)==2:
            # Two-stages are assumed
            self.__s = (int_array( value[0], dim=p, down=False ),\
                        int_array( value[1], dim=n, down=False ) )
            self.__stages = 2
        else:
            # Only one stage is assumed
            self.__stages = 1 
            self.__two_level = False 
            self.__s = int_array( value, dim=n, down=False )
            
        self.__check_if_uncap()
        
    @property 
    def f(self):
        """Returns the array(s) of facility fixed costs"""
        return self.__f 
     
    @f.setter 
    def f(self, value ):
        """
        Sets array(s) of facility fixed costs to the given value.
        See pyloa.util.int_array for more information. 
        In case of a two-level problem, value must be a 2-tuple
        that specifies the fixed costs for both levels of facilities. 
        """
        if not self.__f is None: 
            if isinstance(self.__f,tuple):
                p, n = len(self.__f[0]), len(self.__f[1])
            else:
                n = len(self.__f) 
        elif not self.__s is None:
            if isinstance( self.__s, tuple ):
                p, n = len(self.__s[0]), len(self.__s[1]) 
            else:
                n = len(self.__s) 
        else:
            p, n = 0, 0
        
        if isinstance(value,tuple) and len(value)==2:
            # Two-level problem is assumed
            self.__stages = 2
            self.__two_level = True   
            self.__s = (real_array(value[0], p),real_array(value[1], n))
        else:
            self.__f = real_array( value, n )
            self.__stages = 1
            self.__two_level = False
            
    @property 
    def get_fcost(self ):
        """Return the fixed cost of the open depots. For a two-level problem,
        a 2-tuple of the fixed cost on both stages is returned."""
        if self.facilities is None: return 0.0 
        if self.__stages==1 or not self.__two_level:      
            return float(self.__f[self.facilities].sum())
        else: 
            f0 = 0.0 if self.facilities[0] is None else float(self.f[0][self.facilities[0]].sum())
            f1 = 0.0 if self.facilities[1] is None else float(self.f[1][self.facilities[1]].sum())
            return( f0, f1 )   
    
    @property 
    def stages(self):
        """
        Return the number of stages (levels) in the current problem instance
        """                
        return self.__stages 
    
    @property 
    def two_level(self):
        """
        Return True if the current problem instance is a two-level problem.
        """                
        return self.__two_level
    
    @property 
    def unitCost(self):
        """
        Return True if cost to supply customers are costs per unit.
        Otherwise, they are cost to supply all of a customer's demand.
        """
        return self.__unitCost
    
    @property
    def uncap(self):
        """Respectively returns True and (True,True) is the facilities
        are uncapacitated."""
        return self.__uncap 
        
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
        Location binary variables z[i] for 
        i=0,...,p-1 and p number of plant sites.
        """
        return self.__z 
    
    @z.setter 
    def z(self,value):
        self.__z = value  

    @property 
    def v(self):
        """
        Flow variables v[i,j] for i=0,...,p-1 
        and j=0,...,n-1 where p is the number of 
        plant sites and n the number of facility sites.
        """
        return self.__v
    
    @v.setter 
    def v(self,value):
        self.__v = value

    @property
    def useBenders(self):
        """Property: Use of Benders' decomposition.""" 
        return self.__useBenders  
    
    @useBenders.setter
    def useBenders( self, value : bool ): 
        if value in (0,1,2):
            self.__useBenders = value
            if value==1 and not self.model is None and not self.model.mipModel is None:
                self.__model.doBenders(3)
    
    @property 
    def addVubs(self):
        """Property: 
            0 if Vubs should not be included, 
            1 if Vubs should be include delayed (user cuts), 
            2 if Vubs should be included
        """ 
        return self.__addVubs 
    
    @addVubs.setter 
    def addVubs(self, value : int ): 
        self.__addVubs = max(0,min(2,value))
    
    @property 
    def timeLim(self):
        """Property: Limit on time a solver may use""" 
        return self.__timeLim 
    
    @timeLim.setter 
    def timeLim(self, value : int ): 
        self.__timeLim = max(0,value)
    
    @property 
    def nodeLim(self):
        """Property: Limit on number of nodes the MIP solver may enumerate""" 
        return self.__nodeLim 
    
    @nodeLim.setter 
    def nodeLim(self, value : int ): 
        self.__nodeLim = value 

    @property 
    def optTol(self): 
        """Property: Optimality tolerance used by the MIP solver""" 
        return self.__optTol 
    
    @optTol.setter 
    def optTol(self, value ):
        """Set optimality tolerance to the given value."""
        if value > 0.0: self.__optTol = value  

    @property 
    def useLB(self):
        """Property: useLB (should Cplex employ local branching)""" 
        return self.__useLB
    
    @useLB.setter 
    def useLB(self, value : bool ): 
        """Switch Cplex's local branching heuristic on (True)
        or off (False)"""
        self.__useLB = value
        if not self.__model is None: self.__model.lbHeur = int(value) 
    
    @property 
    def cuts(self):
        """Property: cuts parameter if MIP solver is used:""" 
        return self.__cuts 
    
    @cuts.setter 
    def cuts(self, value):
        """Set cuts parameter. value=-1 is automatic, value=0
        is switched off, value=2 is aggressive cut generation,
        value = 3 is very aggressive cut generation."""
        if value in (-1,0,1,2,3):
            self.__cuts = value 
            if not self.__model is None: self.__model.cuts = value 
               
    @property 
    def mipStrategy(self):
        """Cplex's search strategy:
               0 : automatic
               1 : traditional branch-and-cut
               2 : dynamic search
        """
        return self.__mipStrategy
    
    @mipStrategy.setter 
    def mipStrategy(self, value : int):
        """Set Cplex's mip search strategy"""
        if value in (0,1,2):
            self.__mipStrategy = value 
            if not self.__model is None: 
                self.__model.mipStrategy = value 
         
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

    #---------------------------------------------
    
    def default_options(self):
        """
        Set solver options to default values
        """
        self.__useBenders = 0
        """Usage of Benders. Default is: do not use"""
         
        self.__addVubs = 2
        """Handling of variable upper bounds. Default: Add them.""" 
           
        self.__timeLim = 0
        """Time limit for the solver"""
        
        self.__nodeLim = -1
        """Limit on number of nodes in enumeration tree"""

        self.__optTol = 1.0E-07
        """Optimality tolerance for the MIP solver."""
     
        self.__useLB = False 
        """No use of local branching heuristic"""
    
        self.__cuts = -1
        """Let the solver decice about cuts."""
    
        self.__mipStrategy = 0
        """If Cplex is used, let Cplex decicide about its search strategy."""
             
    #---------------------------------------------
    