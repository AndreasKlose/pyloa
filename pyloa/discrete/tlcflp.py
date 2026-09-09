"""
    Module discrete.tlcflp of package pyloa:
    Methods for solving a two-stage or two-level CFLPs.
"""
import numpy as np
from itertools import product
from pyloa.mip.model import Model
from pyloa.discrete.dflprob import DFLProblem
from pyloa.discrete.cflp import CFLP
from pyloa.discrete.CFLsg import is_ready as cfl_sg_ok 
from pyloa.discrete.CFLsg import solveCFLP 

#--------------------------------------------------------------------------

class VubGenerator:
    """
    Callback class that looks for violated variable upper bound constraints
    :math:`x_{ij} \\le y_j` and :math:`v_{ij} \\le p_i*y_j` and passes them 
    to the solver. We add for each facility j the most violated of these 
    inequalities. 
    """
    def __init__(self, M, p, m, n, s, x, y, v, ucap ):
        """
        Initialize the class with the model, the number of plants, customers and 
        facilities, the plant and depot capacities s, the allocation variables x, 
        location variables y and the flow variables v. Argument ucap is True,
        if the first stage facilities are uncapacitated. In that case there
        are no VUBs involving variables v(i,j) to add.
        """
        self.M = M 
        self.p = p 
        self.m = m 
        self.n = n
        self.s = s
        self.x = x 
        self.y = y 
        self.v = None if ucap else v 
        
    def __findVubs(self):
        
        yv = np.array( self.M.cbGetNodeRel( self.y.values() ) )
        fy = np.where ( (yv > self.M.IntFeasTol) & (yv < 1.0-self.M.IntFeasTol) )[0]       
        for j in fy: 
            xv = self.M.cbGetNodeRealKeys( self.x, ((j,k) for k in range(self.m)) )
            k = xv.argmax( )
            if xv[k] - yv[j] > self.M.FeasibilityTol: 
                self.M.cbCut( self.x[j,k]-self.y[j] <= 0.0 )
        
        if not self.v is None:
            # self.v is None if first stage facilities are uncapacitated -> No Vubs    
            for j in fy: 
                vv = self.M.cbGetNodeRealKeys( self.v, ((i,j) for i in range(self.p)) )
                i = np.argmax( vv - self.s[0]*yv[j] ) 
                if vv[i] - self.s[0][i]*yv[j] > self.M.FeasibilityTol:
                    self.M.cbCut( self.v[i,j]-self.s[0][i]*self.y[j] <= 0.0 )
        
              
    def invoke(self, context):
        """
        Method called by Cplex's solver using the single argument cplex.callbacks.Context.
        """
        if self.M.in_relaxation( context ):
            self.M.context = context 
            self.__findVubs()
        
    def __call__( self, model, where ):
        """
        GuRoBi will call this function as callback.
        """
        if self.M.in_relaxation( where ):
            self.__findVubs()
      
#--------------------------------------------------------------------------

class MCFVubGen:
    """
    Callback class that looks for violated variable upper bound constraints
    :math:`\\sum_i x_{ijk} \\le y_j` and :math:`\\sum_j x_{ijk} \\le z_i` for
    for the case of the TLCFLP's multi-commodity formulation. We add for each 
    facility i and j the most violated of these inequalities. 
    """
    def __init__(self, M, p, m, n, x, y, z ):
        """
        Initialize the class with the model, the number of plants, customers and 
        facilities, the allocation variables x, and location variables y and z.
        """
        self.M = M 
        self.p = p 
        self.m = m 
        self.n = n
        self.x = x 
        self.y = y 
        self.z = z 
      
    def __findVubs(self):
        
        p, n, m = self.p, self.n, self.m 
        yv = self.M.cbGetNodeRel( self.y.values() ) 
        fy = np.where ( (yv > self.M.IntFeasTol) & (yv < 1.0-self.M.IntFeasTol) )[0]
        zv = self.M.cbGetNodeRel( self.z.values() )
        fz = np.where ( (zv > self.M.IntFeasTol) & (zv < 1.0-self.M.IntFeasTol) )[0]
        
        # Function to compute flow via depot j to all customers k
        get_flowj = lambda j: np.fromiter(map(lambda k: self.M.cbGetNodeRealKeys(self.x,\
                                    ((i,j,k) for i in range(p))).sum(), range(m)), dtype=float)
        
        # Add violated Vubs \\sum_i x(i,j,k) <= y(j) for some j,k    
        for j in fy:
            flow_j = get_flowj(j) 
            k = flow_j.argmax( )
            if flow_j[k] - yv[j] > self.M.FeasibilityTol: 
                self.M.cbCut( self.M.sum(self.x[i,j,k] for i in range(p))-self.y[j] <= 0.0 )
        
        # Function to compute flow from plant i to all customers k
        get_flowi = lambda i: np.fromiter(map(lambda k: self.M.cbGetNodeRealKeys(self.x,\
                                    ((i,j,k) for j in range(n))).sum(), range(m)), dtype=float) 
        
        # Add violated Vubs \\sum_j x(i,j,k) <= z(i)    
        for i in fz:
            flow_i = get_flowi( i ) 
            k = flow_i.argmax( )
            if flow_i[k] - zv[i] > self.M.FeasibilityTol: 
                self.M.cbCut( self.M.sum(self.x[i,j,k] for j in range(self.n))-self.z[i] <= 0.0 )
              
    def invoke(self, context):
        """
        Method called by Cplex's solver using the single argument cplex.callbacks.Context
        """
        if self.M.in_relaxation( context ):
            self.M.context = context 
            self.__findVubs()
        
    def __call__( self, model, where ):
        """
        GuRoBi will call this function as callback.
        """
        if self.M.in_relaxation( where ):
            self.__findVubs()
      
#--------------------------------------------------------------------------

class TLCFLP( DFLProblem ):
    """
    Class implementing methods for solving the two-stage or two-level
    (un-)capacitated facility location problem.
    """
 
    def __init__( self, fname=None, formt='TSCFLP', unitCost=True, scale=1, mcf=False, \
                  d=None, s=None, f=None, c=None ):
        """
        Creates instance of a two-stage or two-level CFLP. 
        
        See base class 'DFLProblem' in module dflprob.
        """    
        super().__init__(fname=fname, formt=formt, unitCost=unitCost, scale=scale, mcf=mcf,\
                          d=d, s=s, f=f, c=c )
        
        self.__mc_form = self.two_level and isinstance(self.c, np.ndarray) and self.c.ndim==3
        """True for a two-level problem using the multi-commodity flow formulation."""
        
        # If the multi-commodity formulation is not used, then demand values must
        # be present as otherwise the cost for distribution on the first stage from
        # plants to depots cannot be obtained.
        assert ( self.__mc_form or not self.d is None ), \
                 "For computing distributon cost on 1st stage demand data must be defined!"  
        
        self.__cflp_solver = 'mip'
        """Default CFlP solver is a general MILP solver. Alternatively, 'sg' can be 
        selected."""
        
        self.__cfl = None
        """Handle to the CFLP instance to be solved within the heuristic(s)."""
        
        self.__mcf = None 
        """Handle to the min-cost network flow problem to be solved for given facility sites."""
       
        self.__dep_cap = None 
        """Pointer to depot capacity constraints (only used in the heuristic)."""
        
        self.__plant_cap = None 
        """Pointer to plant capacity constraints (only used in the heuristic)."""
        
        self.__flow_c = None 
        """Pointer to flow conversation constraints (only used in the heuristic)."""
        
        # Set MIP solver options to default values.
        self.default_options()
        
    #---------------------------------------------
    # Class properties
    #---------------------------------------------
       
    @property 
    def cflp_solver(self):
        """Property: Solver to be used for an underlying CFLP. Either Cplex or SG""" 
        return self.__cflp_solver  
    
    @cflp_solver.setter 
    def cflp_solver(self, value : str ): 
        self.__cflp_solver = value
      
    #---------------------------------------------
    
    def __is_uncap(self):
        """
        Check if plant capacity constraints in a TSCFLP 
        are redundant.
        """
        source = np.argmin( self.c[0], axis=0 )
        pcap = self.s[0]
        ucap0 = not any( pcap[i] < self.totalDemand for i in source ) 
        if ucap0: self.assigned = (list(source), None )
        return ucap0
    
    #---------------------------------------------
    
    def __cfl_solver_options(self, cfl ):
        """
        Set the options for the CFLP solver to the same
        values as set for this TSCFLP/TLCFLP instance.
        
        Parameters
        ----------
        cfl : class CFLP
            Instance of the CFLP to solve
        """
        cfl.solver = self.__cflp_solver 
        cfl.nodeLim = self.nodeLim 
        cfl.timeLim = self.timeLim
        cfl.cuts = self.cuts 
        cfl.mipStrategy = self.mipStrategy 
        cfl.useLB = self.useLB 
        cfl.useBenders = self.useBenders
        cfl.addVubs = self.addVubs 
        cfl.optTol = self.optTol
        cfl.silent = self.silent 
    
    #---------------------------------------------
    
    def __solve_as_CFLP( self ):
        """
        If the plant capacities are actually redundant,
        we can solve the TSCFLP as an ordinary CFLP.
        """
        T = self.c[0]   # Unit cost of transport plants -> depots
        _, n = T.shape  # Number of plants and depot sites
        
        C = self.c[1].copy() # Unit or total cost to supply each customer 
        m = C.shape[1]
        
        source = self.assigned[0] # Plant supplying each depot if open
        
        if self.unitCost:
            for j,i  in enumerate(source): C[j] += T[i,j]
        else:
            for j,i in enumerate(source): C[j] += T[i,j]*self.d
        
        # Create and solve CFLP instance
        cfl = CFLP( unitCost=self.unitCost, d=self.d, s=self.s[1], f=self.f, c=C )
        self.__cfl_solver_options(cfl)
    
        # Solve the CFLP
        cfl.solve( get_flows=True )
        
        # Extract the solution
        self.facilities = cfl.facilities
        self.cost = cfl.cost
        self.bound = cfl.bound
        self.fcost = cfl.fcost
        sup2 = cfl.supply if cfl.assigned is None else \
               dict.fromkeys( zip(cfl.assigned,range(m)), 1.0  )
        throughput = np.zeros(n, dtype=float)
        for j,k in sup2: throughput[j] += self.d[k]*sup2[(j,k)] 
        sup1 = dict( zip(zip(source,range(n)),throughput) )
        tc_0_1 = sum( T[i,j]*sup1[(i,j)] for i,j in sup1 )
        self.supply = (sup1, sup2 )
        self.tcost = ( tc_0_1, cfl.tcost - tc_0_1 )
      
        self.ctime = cfl.ctime 
        self.mip_time = cfl.mip_time 
        self.mip_work = cfl.mip_work  
        self.nodeCount = cfl.nodeCount 
        
    #---------------------------------------------
      
    def __tscflp(self):
        """
        Model the TSCFLP 
        """
        d, s, f, t, c = self.d, self.s, self.f, self.c[0], self.c[1]
        p, _ = t.shape 
        n, m = c.shape
            
        # Create the model
        M = Model('TSCFLP')
        
        # Create the binary and continuous flow variables
        y = M.addVars( n, vtype='B' )
        x = M.addVarMatrix( n, m )
        v = M.addVarMatrix( p, n )
    
        # Objective function is to minimize total cost
        fc  = M.sum( f[j]*y[j] for j in range(n) )
        tc1 = M.sum( c[j,k]*d[k]*x[j,k] for j,k in x.keys() ) if self.unitCost else\
              M.sum( c[j,k]*x[j,k] for j,k in x.keys() )
        tc0 = M.sum( t[i,j]*v[i,j] for i,j in v.keys() )
        M.minimize( tc0 + tc1 + fc )  
    
        # Add the demand constraints
        M.addConstraints( M.sum( x[j,i] for j in range(n) ) == 1 for i in range(m) )
        
        # Add the flow conversation constraints
        M.addConstraints( M.sum(v[i,j] for i in range(p)) == M.sum(x[j,k]*d[k] for k in range(m)) \
                          for j in range(n) ) 
        
        # Add capacity constraints on stage 2
        if self.uncap[1]:
            if self.addVubs < 2:
                M.addConstraints( M.sum(x[j,i] for i in range(m)) <= (m-1)*y[j] for j in range(n) )
        else: 
            M.addConstraints( M.sum( d[i]*x[j,i] for i in range(m)) <= s[1][j]*y[j] for j in range(n) )
            M.addConstraint( M.sum( s[1][j]*y[j] for j in range(n) ) >= self.totalDemand )
            
        # Add capacity constraints on stage 1 
        M.addConstraints( M.sum( v[i,j] for j in range(n) ) <= s[0][i] for i in range(p) ) 
    
        # Include variable upper bounds as far as desired
        if self.addVubs > 1:
            M.addConstraints( x[j,k]-y[j] <= 0 for j,k in x.keys() )
            if self.uncap[1]:
                # No depot capacities
                M.addConstraints( v[i,j]-s[0][i]*y[j] <=0 for i,j in v.keys() )
            else: 
                M.addConstraints( v[i,j]-s[0][i]*y[j] <=0 for i,j in v.keys() \
                                  if s[0][i] < s[1][j] )
        elif self.addVubs > 0:
            vubgen = VubGenerator( M, p, m, n, s, x, y, v, True )
            M.set_callback( vubgen, contxtmsk = M.id_relaxation )
        
        self.model = M
        self.x = x 
        self.y = y
        self.v = v
    
    #---------------------------------------------
      
    def __tlcflp_2index(self):
        """
        Model the TLCFLP using 2-index variables for the flows.
        """
        d, s, f, t, c = self.d, self.s, self.f, self.c[0], self.c[1]
        p, _ = t.shape 
        n, m = c.shape
        totD = self.totalDemand
            
        # Create the model
        M = Model('2TLCFLP')
        
        # Create the binary and continuous flow variables
        y = M.addVars( n, vtype='B' )
        z = M.addVars( p, vtype='B' )
        x = M.addVarMatrix( n, m )
        v = M.addVarMatrix( p, n )
    
        # Objective function is to minimize total cost
        fc0 = M.sum( f[0][i]*z[i] for i in range(p) )
        fc1 = M.sum( f[1][j]*y[j] for j in range(n) )
        tc0 = M.sum( t[i,j]*v[i,j] for i,j in v.keys() )
        tc1 = M.sum( c[j,k]*d[k]*x[j,k] for j,k in x.keys() ) if self.unitCost else \
              M.sum( c[j,k]*x[j,k] for j,k in x.keys() )
        M.minimize( fc0 + fc1 + tc0 + tc1 )
     
        # Add the demand constraints
        M.addConstraints( M.sum( x[j,k] for j in range(n) ) == 1 for k in range(m) )
        
        # Add the flow conversation constraints
        M.addConstraints( M.sum(v[i,j] for i in range(p)) == M.sum(x[j,k]*d[k] for k in range(m)) \
                            for j in range(n) ) 
        
        # Add capacity constraints on stage 2
        if self.uncap[1]:
            if self.addVubs < 2:
                M.addConstraints( M.sum(x[j,k] for k in range(m)) <= m*y[j] for j in range(n) )
        else: 
            M.addConstraints( M.sum(d[k]*x[j,k] for k in range(m)) <= s[1][j]*y[j] for j in range(n) )
            M.addConstraint( M.sum( s[1][j]*y[j] for j in range(n)) >= totD )
            
        # Add capacity constraints on stage 1
        if self.uncap[0]:
            M.addConstraints( M.sum(v[i,j] for j in range(n)) <= totD*z[i] for i in range(p) )
        else:
            M.addConstraints( M.sum( v[i,j] for j in range(n) ) <= s[0][i]*z[i] for i in range(p) ) 
            M.addConstraint( M.sum( s[0][i]*z[i] for i in range(p) ) >= totD )
    
        # Include variable upper bounds as far as desired
        if self.addVubs > 1:
            M.addConstraints( x[j,k]-y[j] <= 0 for j,k in x.keys() )
            if not self.uncap[0]:  
                M.addConstraints( v[i,j]-s[0][i]*y[j] <= 0 for i,j in v.keys() if s[0][i] < s[1][j] )
        elif self.addVubs > 0:
            vubgen = VubGenerator( M, p, m, n, s, x, y, v, self.uncap[0] )
            M.set_callback( vubgen, contxtmsk = M.id_relaxation )
               
        self.model = M
        self.x = x 
        self.y = y
        self.v = v
        self.z = z
    
    #---------------------------------------------
      
    def __tlflp_mc_form(self):
        """
        Model the TLCFLP using the multi-commodity formulation
        (3-index flow variables)
        """
        d, s, f, c = self.d, self.s, self.f, self.c
        p, n, m = c.shape 
            
        # Create the model
        M = Model('TLCFLP-MC')
        
        # Create the binary and continuous flow variables
        y = M.addVars( n, vtype='B' )
        z = M.addVars( p, vtype='B' )
        x = M.addVarCube( (p, n, m) )
        
        # Objective function is to minimize total cost
        fc0 = M.sum( f[0][i]*z[i] for i in range(p) )
        fc1 = M.sum( f[1][j]*y[j] for j in range(n) )
        tc  = M.sum( c[i,j,k]*d[k]*x[i,j,k] for i,j,k in x.keys() ) if self.unitCost else \
              M.sum( c[i,j,k]*x[i,j,k] for i,j,k in x.keys() )
        M.minimize( fc0 + fc1 + tc )
    
        # Add the demand constraints
        M.addConstraints( M.sum( x[i,j,k] for i,j in product(range(p),range(n)) ) == 1 for k in range(m) )
        
        # Add capacity constraints on stage 1
        totD = self.totalDemand
        if not self.uncap[0]:
            M.addConstraints( M.sum( d[k]*x[i,j,k] for j,k in product(range(n),range(m)) ) <= s[0][i]*z[i] \
                              for i in range(p) )
            M.addConstraint( M.sum( s[0][i]*z[i] for i in range(p) ) >= totD )
        elif self.addVubs < 2:
            M.addConstraints( M.sum( x[i,j,k] for j,k in product(range(n),range(m)) ) <= m*z[i] \
                              for i in range(p) )
            
        # Add capacity constraints on stage 2
        if not self.uncap[1]:
            M.addConstraints( M.sum( d[k]*x[i,j,k] for i,k in product(range(p),range(m)) ) <= s[1][j]*y[j] \
                              for j in range(n) )
            M.addConstraint( M.sum( s[1][j]*y[j] for j in range(n) ) >= totD )
        elif self.addVubs < 2: 
            M.addConstraints( M.sum( x[i,j,k] for i,k in product(range(p),range(m)) ) <= m*y[j] \
                              for j in range(n) )
                
        # Include variable upper bounds as far as desired
        if self.addVubs > 1:
            M.addConstraints( M.sum( x[i,j,k] for i in range(p) ) - y[j] <= 0 \
                              for j,k in product(range(n),range(m)) )
            M.addConstraints( M.sum( x[i,j,k] for j in range(n) ) - z[i] <= 0 \
                              for i,k in product(range(p),range(m)) ) 
        elif self.addVubs > 0:
            vubgen = MCFVubGen( M, p, m, n, x, y, z )
            M.set_callback( vubgen, contxtmsk = M.id_relaxation )
     
        self.model = M
        self.x = x 
        self.y = y
        self.z = z
    
    #---------------------------------------------
    
    def __MIP_options ( self ):
        """
        Pass options to the MIP solver
        """ 
        M = self.model
        M.timelimit = self.timeLim  # Time limit 
        M.nodelimit = self.nodeLim  # Node limit 
        M.lbHeur = self.useLB       # Use of local branching?
        M.mipEmphasis = 2           # Focus on optimality
        M.MIPGap = self.optTol      # Relative optimality tolerance 
        M.cuts = self.cuts          # Usage of cuts
        # Search strategy in case Cplex is used
        M.mipStrategy = self.mipStrategy
        # If Cplex is used, activate automatic Benders if desired
        if self.useBenders > 0: M.doBenders = 3
    
    #---------------------------------------------
    
    def __get_mip_solution( self ):
        """Extract solution from the MIP solver"""
        M = self.model
        if self.two_level:
            self.facilities = (list(M.get_solution(self.z,keep_zeros=False,precision=0.1).keys()),\
                               list(M.get_solution(self.y,keep_zeros=False,precision=0.1).keys()))
        else: 
            self.facilities = list(M.get_solution( self.y, keep_zeros=False, precision=0.1).keys() )
        
        self.fcost = self.get_fcost*self._c_scale
        self.cost = M.ObjVal*self._c_scale  
        self.bound = M.ObjBound*self._c_scale
        self.cost += self.ccnst 
        self.bound += self.ccnst
        
        mcf = self.__mc_form
        if mcf: 
            self.supply = M.get_solution( self.x, keep_zeros=False )
            self.tcost = self.cost - self.fcost[0] - self.fcost[1]
        else:
            self.supply = (M.get_solution(self.v, keep_zeros=False), M.get_solution(self.x, keep_zeros=False))
            tc_0_1 = sum( self.c[0][i,j]*self.supply[0][(i,j)] for i,j in self.supply[0].keys() )
            tc_1_2 = sum( self.c[1][j,k]*self.supply[1][(j,k)] for j,k in self.supply[1].keys() ) + self.ccnst 
            self.tcost = (float(tc_0_1), float(tc_1_2) )
    
    #---------------------------------------------
    
    def solve( self, keep_model=False ):
        """
        Solve the instance of a two-stage or two-level capacitated 
        facility location problem.
        
        Parameters
        ----------
        keep_model: bool, optional 
            If True, the created instance of docplex.mp.model.Model is not
            destroyed but kept. 
        """ 
        # If uncapacitated first stage and not a two-level problem, solve it as CFLP
        if not self.two_level and self.__is_uncap():
            # Uses same time and node limit as defined for this TSCFLP instance
            self.__solve_as_CFLP( )
            return 
        
        no_scale =  not self._c_is_scaled
        if no_scale: self.scale_costs( )
        
        if self.two_level:
            # Two-level problem
            if self.__mc_form:
                self.__tlflp_mc_form( )
            else:
                self.__tlcflp_2index( )
        else:
            # Two-stage problem (location decisions only on 2nd stage)
            self.__tscflp( )
       
        # Set solver options and start time
        self.__MIP_options()
        self._set_starttime()
        
        # Solve MILP
        M = self.model 
        M.log_output = not self.silent
        if M.optim( ):
            self._set_comptime( mip_time=True )
            self.__get_mip_solution( ) 
        if not keep_model: self.model.end()
            
        # Undo scaling of cost
        if no_scale: self.unscale_costs()
    
    #---------------------------------------------
    
    def __model_mcf(self):
        """
        Set up and return the mathematical model of the
        min-cost network flow problem for given plant sites
        via given depot sites to the customers. 
        """
        d, s, t, c = self.d, self.s, self.c[0], self.c[1]
        p, _ = t.shape 
        n, m = c.shape
            
        # Create the model
        M = Model('MCF')
        
        # Create the binary and continuous flow variables
        x = M.addVarMatrix( n, m ) # Flow from depots to customers
        v = M.addVarMatrix( p, n ) # Flow from plants to depots
    
        # Objective function is to minimize total cost
        tc0 = M.sum( t[i,j]*v[i,j] for i,j in v.keys() )
        tc1 = M.sum( c[j,k]*x[j,k] for j,k in x.keys() ) if self.unitCost else \
              M.sum( c[j,k]/d[k]*x[j,k] for j,k in x.keys() )
        M.minimize( tc0 + tc1 )
     
        # Add the demand constraints
        M.addConstraints( M.sum( x[j,k] for j in range(n) ) == d[k] for k in range(m) )
        
        # Add the flow conversation constraints
        flow_c = M.addConstraints( (M.sum(v[i,j] for i in range(p)) == M.sum(x[j,k] for k in range(m)) \
                                   for j in range(n)), return_constr=self.two_level ) 
        
        # Add depot capacity constraints 
        dep_cap = M.addConstraints( (M.sum(x[j,k] for k in range(m)) <= s[1][j] for j in range(n)),\
                                     return_constr=True )
            
        # Add plant capacity constraints
        plant_cap = M.addConstraints( (M.sum( v[i,j] for j in range(n) ) <= s[0][i] \
                                      for i in range(p)), return_constr=True )
    
        self.x, self.v = x, v 
        self.__dep_cap, self.__plant_cap, self.__flow_c = dep_cap, plant_cap, flow_c

        M.log_output = False
        self.__mcf = M
     
    #---------------------------------------------
    
    def __adjust_mcf( self, facilities ):
        """
        Adjust the capacity constraints in the min-cost
        network flow problem by resetting the capacity
        of the open/closed facilities.
        """
        M = self.__mcf 
        p, n = self.c[0].shape 
        S0, S1 = facilities if self.two_level else (None, facilities) 
        
        if self.two_level:
            # Adjust plant capacity constraints
            is_open = np.zeros(p, dtype=bool)
            is_open[S0] = True
            for i, cnstr in enumerate(self.__plant_cap):
                M.set_rhs(cnstr, self.s[0][i]*is_open[i])
        
        # Adjust depot capacity constraints
        is_open = np.zeros(n, dtype=bool)
        is_open[S1] = True 
        for j, cnstr in enumerate( self.__dep_cap ):
            M.set_rhs(cnstr, self.s[1][j]*is_open[j]) 
    
    #---------------------------------------------

    def get_full_solution( self, facilities ):
        """
        Computing the min-cost network flow and compute all
        cost components to complete the partial solution given
        by the set of open facilities.

        Parameters
        ----------
        facilities : list of int or 2-tuple of such lists
            List of open depots in case of a TSCFLP and a tuple
            consisting of the list of open plants and the
            list of open depots in case of a TLCFLP.
        """
        mcf = self.__mcf 
        if mcf is None: self.__model_mcf()
        self.__adjust_mcf( facilities )
        self.facilities = facilities
        mcf.optim()
        self.fcost = self.get_fcost
        tot_fcost = self.fcost[0] + self.fcost[1] if self.two_level else self.fcost
        self.cost = tot_fcost + mcf.ObjVal + self.ccnst
        self.supply = (mcf.get_solution(self.v, keep_zeros=False),\
                       mcf.get_solution(self.x, keep_zeros=False) )
        tc_0_1 = sum( self.c[0][i,j]*self.supply[0][(i,j)] for i,j in self.supply[0].keys() )
        if self.unitCost:
            tc_1_2 = sum( self.c[1][j,k]*self.supply[1][(j,k)] for j,k in self.supply[1].keys() )
            for j,k in self.supply[1]: self.supply[1][(j,k)] /= self.d[k]
        else: 
            for j,k in self.supply[1]: self.supply[1][(j,k)] /= self.d[k]
            tc_1_2 = sum( self.c[1][j,k]*self.supply[1][(j,k)] for j,k in self.supply[1].keys() )
        self.tcost = (float(tc_0_1), float(tc_1_2) + self.ccnst )

    #---------------------------------------------

    def __sol_exists( self, S, solDict ):
        """
        Checks if set S of open facilities is already 
        included in the dictionary solDict of sets of 
        open facilities or not. If not, the set S is 
        added to the dict.
        """
        key = (len(S),sum(S))
        SList = solDict.get(key)
        if SList is None:
            solDict[key] = list([S])
            return False 
        for T in SList: 
            if np.all( S==T ): return True 
        solDict[key].append(S)
        return False
    
    #---------------------------------------------
    
    def __init_CFLP(self):
        """
        Initialize the CFLP to be solved as a subproblem
        within each iteration of the heuristic.
        """
        # Costs to serve all of a customer's demand from the depots
        C = self.c[1].copy() 
        if self.unitCost: C *= self.d
        # Create CFLP instance        
        f = self.f[1] if self.two_level else self.f
        cfl = CFLP( unitCost=False, d=self.d, s=self.s[1], f=f, c=C )
        self.__cfl_solver_options(cfl)
        cfl.silent = True
        if self.__cflp_solver=='sg' and  not cfl_sg_ok: self.__cflp_solver='mip'
        self.__cfl = cfl 

    #---------------------------------------------
    
    def __cflp_lrsub(self, prices):
        """
        Solve that part of the Lagrangian subproblem that
        is a CFLP and results if in the TSCFLP the plant capacity 
        and in the TLCFLP the flow conversation constraints are 
        relaxed in a Lagrangian manner. Return the lower bound obtained 
        from that and the set of open depots in the obtained solution.

        Parameter
        --------
        prices : list of float
             The list of the (negative) of the dual prices.
        """
        cfl = self.__cfl 
        C = cfl.c
        if self.two_level:
            # TLCFLP: Relaxation of flow conversation constraints.
            for j, c in enumerate(C): c += prices[j]*self.d
        else:
            # TSCFLP: Relaxation of plant capacities.
            # Obtain the cheapest source for each depot if it is open.
            T = self.c[0] 
            source = np.argmin( T - np.vstack(prices), axis=0 )
            # Adjust the cost in the CFLP instance.
            for j, i in enumerate(source): C[j] += (T[i,j]-prices[i])*self.d
      
        # Solve the CFLP instance
        lowbnd = 0.0
        if self.__cflp_solver == 'sg':
            cfl._cmin = C.min( axis=0 )
            C -= cfl._cmin
            cfl.ccnst = cfl._cmin.sum()
            status, cfl.cost, cfl.facilities, _ = solveCFLP( cfl.d, cfl.s, cfl.f, C,\
                                    nodeLim=cfl.nodeLim, screenOn=0)
            C += cfl._cmin
            if status >= 70000: lowbnd = cfl.cost + cfl.ccnst
        else:
            M = cfl.model 
            if M is None:
                cfl.solve(keep_model=True, get_flows=False)
                lowbnd = cfl.bound
            else:
                fc = M.sum( cfl.f[j]*cfl.y[j] for j in cfl.y )
                tc = M.sum( C[j,k]*cfl.x[j,k] for j,k in cfl.x )
                M.minimize( fc + tc )
                if M.optim():
                    cfl.facilities = list(M.get_solution( cfl.y, keep_zeros=False, precision=0.1).keys() )
                    lowbnd = M.ObjBound

        # Restore original cost matrix and adjust lower bound for constant term
        if self.two_level:
            for j, c in enumerate(C) : c -= prices[j]*self.d
        else:
            for j, i in enumerate(source): C[j] -= (T[i,j]-prices[i])*self.d
            lowbnd += np.dot(prices, self.s[0])
    
        return lowbnd
        
    #---------------------------------------------
    
    def __tscflp_heur_cd( self, min_iter, max_iter ):
        """
        Heuristic for the TSCFLP relying on a Lagrangian
        relaxation of the plant capacity constraints.
        The resulting Lagrangian subproblem is a CFLP.
        Lagrangian multipliers are generated as in the
        cross decomposition method's subproblem phase.
        When obtaining a set of open depots from the
        solution to the subproblem, the min-cost network
        flow problem is solved and the node potentials
        of the plant nodes used as Lagrangian multipliers.
        """
        # Best set of open facilities, current best lower and upper bound
        Sbest, lowbnd, upbnd = None, 0.0, float('inf')
        
        # Dictionary of sets of open depots generated
        solDict = dict()
        prices = [0.0]*self.c[0].shape[0]
        
        # Set up the model for the min-cost network flow problem
        self.__model_mcf( )
        mcf = self.__mcf
        
        # Initialize the CFLP subproblem
        self.__init_CFLP()
       
        # Iterate between solving a Lagrangian subproblem
        # and a min-cost network flow for obtaining new
        # dual prices.
        itr = 0
        maxiter = np.iinfo(np.int32).max if max_iter==0 else max_iter
        if maxiter < min_iter: maxiter=min_iter
        
        if not self.silent:
            print('='*50)
            print('Lagrangian based heuristic for the TSCFLP')
            print('='*50)
            print("Iter  Lower bound  Upper bound  Best objective")
            print('-'*50)
             
        self._set_starttime()
        while True:
            itr += 1
            if not self.silent: print(f"{itr:4d}",end='  ')
            curbnd = self.__cflp_lrsub( prices )
            improve = curbnd > lowbnd
            if improve: lowbnd = curbnd
            S = self.__cfl.facilities
            # Stop if set of S of facilities already generated
            if self.__sol_exists( S, solDict ): 
                if not self.silent: print('No new set of facilities generated')
                break
            self.__adjust_mcf( S )
            if not mcf.optim(): break 
            cur_obj = mcf.ObjVal + self.f[S].sum()
            prices = mcf.get_dual_values(self.__plant_cap)
            if cur_obj < upbnd: 
                improve = True
                Sbest = self.__cfl.facilities
                upbnd = cur_obj
            if not self.silent:
                print(f"{curbnd:11.2f}  {cur_obj:11.2f}  {upbnd:14.2f}") 
            # Stop in case that solution is proven optimal   
            if (upbnd - lowbnd)/max(1.0,lowbnd) < 1.0E-04: break 
            # Stop if no improve in lower or upper bound or maximal iterations reached
            if (not improve or itr==maxiter) and itr >= min_iter: break
            
        if not self.silent: print('-'*50)

        # Reset the solution to the best one found above
        self.get_full_solution( Sbest )
        self.bound = lowbnd + self.ccnst      
        self._set_comptime()
        mcf.end()
        if not self.__cfl.model is None: self.__cfl.model.end()
    
    #---------------------------------------------
    
    def __init_KPsub(self):
        """
        Initialize the binary knapsack problem
        to be solved within the Lagrangian subproblem.
        """
        p, s = self.c[0].shape[0], self.s[0]
        KP = Model('KP-sub')
        z = KP.addVars( p, vtype='B' )
        KP.addConstraint( KP.sum( s[i]*z[i] for i in range(p) ) >= self.totalDemand )
        KP.z = z

        return KP 
    
    #---------------------------------------------
    
    def __LRredcst( self, i, prices ):
        """
        Compute the Lagrangian reduced costs for 
        plant i given the Lagrangian multiplier
        vector 'prices' for the flow conversation
        constraints.
        """
        plt_cap, F, s, profit = self.s[0][i], self.f[0][i],  self.s[1], prices - self.c[0][i]
        positive = np.where( profit > 0.0)[0]
        if len(positive)==0: return F
        positive = sorted( positive, key = lambda j: profit[j], reverse=True ) 
        supply, redcost = 0, F
        for j in positive:
            cap = min( plt_cap, s[j] )
            if supply + cap >= plt_cap:
                redcost -= profit[j]*(plt_cap-supply)
                break
            supply += cap
            redcost -= profit[j]*cap
        return redcost
                       
    #---------------------------------------------
    
    def __kp_lrsub( self, KP, prices ): 
        """
        Solve the knapsack part of the Lagrangian
        subproblem if in the TLCFLP flow conversation
        constraints are relaxed.
        """
        p = self.c[0].shape[0]
        # Set objective function
        KP.minimize( KP.sum(self.__LRredcst(i,prices)*KP.z[i] for i in range(p)) )
        # Solve binary knapsack problem and obtain open plants
        if KP.optim():
            plants = list(KP.get_solution( KP.z, keep_zeros=False, precision=0.1).keys() )
            return KP.ObjBound, plants
        return 0.0, list(range(p))

    #---------------------------------------------
    
    def __tlcflp_heur_cd(self, min_iter, max_iter):
        """
        Heuristic for the TLCFLP relying on a Lagrangian
        relaxation of the flow conversation constraints.
        The resulting Lagrangian subproblem is a CFLP plus
        a sequence of continuous and a single binary knapsack
        problem. Lagrangian multipliers are generated as in 
        the cross decomposition method's subproblem phase.
        When obtaining a set of open plants and depots from the
        solution to the subproblem, the min-cost network
        flow problem is solved and the node potentials
        of the depot nodes used as Lagrangian multipliers.
        """
        # Best set of open facilities, current best lower and upper bound
        Sbest, lowbnd, upbnd = None, 0.0, float('inf')
        
        # Dictionary of generated sets of open plands and depots.
        plantDict, depotDict = dict(), dict()
        # Initial dual prices are zero
        prices = [0.0]*self.c[0].shape[1]
        
        # Set up the model for the min-cost network flow problem
        self.__model_mcf( )
        mcf = self.__mcf
        
        # Initialize the CFLP subproblem
        self.__init_CFLP()

        # Initialize the binary Knapsack subproblem
        KP = self.__init_KPsub()
       
        # Iterate between solving a Lagrangian subproblem
        # and a min-cost network flow for obtaining new
        # dual prices.
        itr = 0
        maxiter = np.iinfo(np.int32).max if max_iter==0 else max_iter
        if maxiter < min_iter: maxiter=min_iter
        
        if not self.silent:
            print('='*50)
            print('Lagrangian based heuristic for the TLCFLP')
            print('='*50)
            print("Iter  Lower bound  Upper bound  Best objective")
            print('-'*50)
             
        self._set_starttime()
        while True:
            itr += 1
            if not self.silent: print(f"{itr:4d}",end='  ')
            bnd1 = self.__cflp_lrsub( prices )
            S1 = self.__cfl.facilities
            bnd2, S0 = self.__kp_lrsub( KP, prices )
            curbnd = bnd1 + bnd2
            improve = curbnd > lowbnd
            if improve: lowbnd = curbnd
            # Stop if set of open plants and depots already generated
            know_plants = self.__sol_exists( S0, plantDict )
            know_depots = self.__sol_exists( S1, depotDict )
            if know_plants and know_depots:
                if not self.silent: print('No new set of facilities generated')
                break
            self.__adjust_mcf( (S0,S1) )
            if not mcf.optim(): break 
            cur_obj = mcf.ObjVal + self.f[0][S0].sum() + self.f[1][S1].sum()
            prices = mcf.get_dual_values(self.__flow_c)
            if cur_obj < upbnd: 
                improve = True
                Sbest = (S0,S1)
                upbnd = cur_obj
            if not self.silent:
                print(f"{curbnd:11.2f}  {cur_obj:11.2f}  {upbnd:14.2f}") 
            # Stop in case that solution is proven optimal   
            if (upbnd - lowbnd)/max(1.0,lowbnd) < 1.0E-04: break 
            # Stop if no improve in lower or upper bound or maximal iterations reached
            if (not improve or itr==maxiter) and itr >= min_iter: break
            
        if not self.silent: print('-'*50)
                 
        # Reset the solution to the best one found above
        self.get_full_solution( Sbest )
        self.bound = lowbnd + self.ccnst
        
        self._set_comptime()
        mcf.end()
        del KP.z 
        KP.end()
        if not self.__cfl.model is None: self.__cfl.model.end()
       
    #---------------------------------------------
    
    def heuristic(self, min_iter=1, max_iter=0 ):
        """
        Applies a Lagrangian heuristic, resembling cross 
        decomposition's subproblem phase, to the TSCFLP
        and TLCFLP.

        For a two-stage problem, the heuristic relies on
        dualizing the plant capacities. The Lagrangian
        subproblem then reduces to a CFLP. Initially,
        Lagrangian multipliers are set to zero and in
        subsequent iterations obtained as the node potentials
        of stage 1 facilities (plants) obtained from solving 
        the min-cost network flow problem for fixed facility
        locations. 
        
        For a two-level problem (location decisions on
        both stages), the heuristic can only be applied
        if the multi-commodity formulation is not used.
        The flow conversation constraints are then relaxed. 
        This leads to a CFLP to be solved for the second 
        stage as well as a sequence of continuous knapsack 
        problems plus a single binary one on the first
        stage. Using then the plant and depot locations
        found this way, a min-cost network flow problem
        is solved for obtaining a complete feasible 
        solution. The node potentials of the stage 2
        facilities (depots) are then used as new multipliers 
        to relax again the flow conversation constraints and
        to repeat the step of solving the Lagrangian subproblem 
        and the min-cost network flow thereafter.
        
        Parameters
        ----------
        min_iter : int, optional
            Minimum number of iterations to be performed.
            Default is min_iter=1, that is actually no
            lower limit.
        max_iter : int, optional
            If equal to 0, the method keeps iterating as 
            long as a not yet generated location solution
            is determined and either the lower or upper
            bound improves. If max_iter is positive, then 
            the iteration is stopped the latest after 
            max_iter iterations.
            
        Remark
        ------
        Note that solving a CFLPs can be time consuming.
        When solving the CFLPs, the number of nodes to
        be enumerated is thus limited to the number set
        in the attribute nodeLim (self.nodeLim) of this
        instance of the class TLCFLP. By default, this
        limit is basically unlimited. But, as the whole
        method is just a heuristic, it can be recommended
        to set this limit to some reasonable figure as,
        e.g., nodeLim=100. Moreover, as solver for the
        CFLPs, a MIPsolver (Cplex or GuRoBi) or a 
        Lagrangian-based branch-and-bound can be used. 
        Which "CFLP solver" should be used is determined 
        by the class property cflp_solver. 
        """
        if self.two_level:
            assert not self.__mc_form, "TLCFLP heuristic cannot be applied to the multi-commodity formulation!"
            self.__tlcflp_heur_cd(min_iter, max_iter)
        else: 
            # We have to solve a TSCFLP.
            if self.__is_uncap():
                # Unlimited plant capacities -> problem reduces to a classic CFLP.
                self.__solve_as_CFLP( )
            else:
                self.__tscflp_heur_cd(min_iter, max_iter)
