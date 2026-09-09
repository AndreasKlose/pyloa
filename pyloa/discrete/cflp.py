"""
    Module discrete.cflp of package pyloa:
    Methods for solving the capacitated facility location problem.
"""
import networkx as nx
from itertools import product
from pyloa.discrete.dflprob import DFLProblem
from pyloa.mip.model import Model

from pyloa.discrete.CFLsg import is_ready as cfl_sg_ok 
from pyloa.discrete.CFLsg import solveCFLP 

#--------------------------------------------------------------------------

class VubGenerator:
    """
    Callback class that looks for violated variable upper bound constraints
    x(i,j) \\le y(j) and passes them to the solver. We add for each
    facility j the most violated of these inequalities.
    """

    def __init__(self, M, m, n, x, y ):
        """
        Initialize the class with the model, the number of customers and facilities,
        the allocation variables x and location variables y.
        """
        self.m = m 
        self.n = n
        self.M = M 
        self.x = x 
        self.y = y 
        
    def __findVubs(self):
           
        yv = self.M.cbGetNodeRel( self.y.values() ) 
        for j in filter(lambda j : self.M.IntFeasTol < yv[j] < 1.0-self.M.IntFeasTol, range(self.n)):
            xv = self.M.cbGetNodeRealKeys( self.x, ((j,i) for i in range(self.m)) ) 
            i = xv.argmax( )
            if xv[i] - yv[j] > self.M.FeasibilityTol: 
                self.M.cbCut( self.x[j,i]-self.y[j] <= 0.0 )
                
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

class CFLP( DFLProblem ):
    """
    Class implementing methods for solving the (un-)capacitated facility
    location problem
    """
 
    def __init__( self, fname=None, formt='CFL-GK', unitCost=True, scale=1, \
                  d=None, s=None, f=None, c=None ):
        """
        Creates instance of a CFLP. 
        
        See base class 'DFLProblem' in module dflprob.
        """
        super().__init__(fname=fname, formt=formt, unitCost=unitCost, scale=scale,\
                          d=d, s=s, f=f, c=c )
        
        self.default_options()
        
        self.__solver = 'mip'
        """Solver used for solving a problem instance. Possible values are
        mip: use a MIP solver and sg: use Lagrangian based B&B."""
        
    #---------------------------------------------
    # Class properties
    #---------------------------------------------
    
    @property 
    def solver(self):
        """Property: Fixes the solver to use, which is either Cplex or SG""" 
        return self.__solver  
    
    @solver.setter 
    def solver(self, value : str ): 
        self.__solver = value
    
    #---------------------------------------------
    
    def __build_model(self):
        """
        Set up the MILP model
        """
        d, s, f, c = self.d, self.s, self.f, self.c
        n, m = c.shape
         
        # Create doCplex model
        M = Model("UFLP") if self.uncap else Model("CFLP") 
    
        # Create the binary and continuous variables
        y = M.addVars( n, vtype='B' )
        x = M.addVarMatrix( n, m ) 
    
        # Objective function is to minimize total cost
        if self.unitCost:
            costij = lambda j,i : c[j,i]*d[i] 
        else:
            costij = lambda j,i: c[j,i]
        fc = M.sum( f[j]*y[j] for j in range(n) )
        tc = M.sum( costij(j,i)*x[j,i] for j,i in x.keys() )
        M.minimize( fc + tc )
      
        # Add the demand constraints
        M.addConstraints( M.sum( x[j,i] for j in range(n) )== 1 for i in range(m) )
        
        # Add capacity constraints
        if self.uncap:
            if self.addVubs < 2:
                M.addConstraints( M.sum(x[j,i] for i in range(m)) <= (m-1)*y[j] for j in range(n) )
        else: 
            M.addConstraints( M.sum(d[i]*x[j,i] for i in range(m)) <= s[j]*y[j] for j in range(n) )
            M.addConstraint ( M.sum( s[j]*y[j] for j in range(n)) >= d.sum() )
    
        # Include variable upper bounds as far as desired
        if self.addVubs > 1:
            M.addConstraints( x[j,i]-y[j] <= 0 for j,i in x.keys() )
        elif self.addVubs > 0:
            # Set up a callback class that takes care of including variable upper bounds
            vubgen = VubGenerator( M, m, n, x, y )
            M.set_callback( vubgen, contxtmsk = M.id_relaxation )
            
        self.model = M
        self.x = x 
        self.y = y  

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
        self.facilities = list(M.get_solution( self.y, keep_zeros=False, precision=0.1).keys() )
        self.fcost = self.get_fcost*self._c_scale
        self.cost = M.ObjVal*self._c_scale  
        self.bound = M.ObjBound*self._c_scale
        self.cost += self.ccnst 
        self.bound += self.ccnst
        self.tcost = self.cost - self.fcost
        if self.uncap: 
            pairs = M.get_solution( self.x, keep_zeros=False, precision=0.1 ).keys()
            self.assigned = [0]*len(pairs)
            for j, i in pairs: self.assigned[i] = j
        else: 
            self.supply = M.get_solution( self.x, keep_zeros=False )
                   
    #---------------------------------------------
    
    def comp_flows(self, facilities=None ):
        """
        Obtain the values of the flow variables x(j,i) for
        a given set of open facilities. This requires 
        that the facilities' total capacity covers the
        total demand. If that is not the case, False 
        is returned and self.supply set to None. Otherwise
        True is returned and the solution stored in the 
        dictionary self.supply. The flow problem is solved
        using networkx.
        """
        self.supply = None 
        if facilities is None: facilities = self.facilities
        if facilities is None or len(facilities)==0: return False 
         
        c, s, d = self.c, self.s, self.d 
        if self.uncap:
            self.assigned = list(map(lambda j : int(facilities[j]),\
                                     c[facilities].argmin(axis=0)) )
            return True 

        D, S = self.totalDemand, s[facilities].sum()  
        if D > S: return False
        
        n, m = self.c.shape
        G = nx.DiGraph()
        
        # Add nodes to the graph including source and sink
        G.add_nodes_from( facilities )            # Open facilities
        G.add_nodes_from([n+i for i in range(m)]) # Customer nodes
        if S > D: G.add_node(n+m)                 # Dummy supplier
        # Facilities' supplies
        nx.set_node_attributes(G, { j: {'demand':-s[j]} for j in facilities } )
        # Customer demands
        nx.set_node_attributes(G, { n+i: {'demand':d[i]} for i in range(m) } )
        # Dummy customer
        if S > D: nx.set_node_attributes(G, {n+m: {'demand': S-D}})
        
        # Edges from facilities to customers
        G.add_edges_from( list(product( facilities, range(n,n+m) ) ) )
        # Edges from facilities to the dummy customer
        if S > D: G.add_edges_from( (j,n+m) for j in facilities )
        
        # Flow costs on the edges. No edge weight means cost of 0.
        # Note that Networkx expects integer unit flow cost
        if self.unitCost:
            tcost = lambda j,i : int(c[j,i]*10000) 
        else:
            tcost = lambda j,i : int(c[j,i]/d[i]*10000)
        nx.set_edge_attributes(G, { (j,n+i): {'weight': tcost(j,i)} for j,i in product(facilities,range(m)) } )
        
        _ , flowDict = nx.network_simplex(G)
        # Facility-Customer edges of positive flow 
        pflow = tuple( filter( lambda ji : flowDict[ji[0]][n+ji[1]] > 0, product(facilities, range(m)) ) )
        flow = map( lambda ji : flowDict[ji[0]][n+ji[1]]/d[ji[1]], pflow )
        self.supply = dict( zip( pflow, flow ) )
        
        return True  
    
    #---------------------------------------------
    
    def solve(self, keep_model=False, get_flows=True ):
        """
        Solve the instance of a (un-)capacitated facility location problem.
        
        Parameters
        ----------
        keep_model: bool, optional 
            If True, the created instance of MIP model is not destroyed but kept.
        get_flows : bool, optional
            Only applies if self.solver = 'sg'. If True, the flow variable values
            are determined and stored in the dictionary self.supply. 
        """ 
        # Scale costs too avoid very large numbers
        no_scale = not self._c_is_scaled
        if no_scale: self.scale_costs( )
        
        method = self.solver 
        if method.lower() == 'mip':
            # Solve with the MIP solver 
            self.__build_model()
            self.__MIP_options()
            self._set_starttime()
            M = self.model 
            M.log_output = not self.silent
            if M.optim( ):
                self._set_comptime( mip_time=True )
                self.__get_mip_solution( ) 
            if not keep_model: self.model.end()
        elif cfl_sg_ok:
            # Solve with Lagrangian relaxation based branch-and-bound
            if self.unitCost: self.c *= self.d 
            self._set_starttime( )
            # Solves the CFLP but does not return the flow variables
            _ , self.cost, self.facilities, TIM = solveCFLP( self.d, self.s, self.f, self.c,\
                                                             nodeLim=self.nodeLim,\
                                                             screenOn=1-int(self.silent) )
            self._set_comptime()
            if get_flows: self.comp_flows()
            self.ctime = (TIM, self.ctime[1])
            self.cost *= self._c_scale
            self.cost += self.ccnst
            self.fcost = self.get_fcost*self._c_scale
            self.tcost = self.cost - self.fcost
            if self.unitCost: self.c /= self.d
        else:
            raise Exception("CFL-SG not available, as library not loaded successfully.")   
        
        # Undo scaling of cost
        if no_scale: self.unscale_costs()
