"""
    Module plane.weber_cg of package pyloa:
    Column generation for solving the multi-source Weber problem.
"""
import numpy as np
from pyloa.util import euclid
from pyloa.plane.loc_alloc import locAlloc, pmedian_heuristic
from pyloa.mip.model import Model
from pyloa.plane.weber import limitedDist

__optTol = 1.0E-04
"""
Optimality tolerance. The column generation is stopped when the
relative gap between lower bound and the master's objective value
is no larger than this value.
"""

#------------------------------------------------------------------------------

class Pricing:
    """
    Class handling the pricing problem
    """
    
    def __init__(self, Y, w, use_mip=True ):
        """
        Initialize the class.
        
        Parameters
        ----------
        Y : mx2 numpy array of float
            Coordinates of the customer points
        w : numpy array of float or int 
            The customer weights.
        use_mip : bool 
            If True, the pricing problem is solved as
            2nd order cone (quadratically constraint MIP)
            using a MIP solver. Otherwise, the limited
            distance Weber problem is solved using Drezner's
            algorithm. 
        """
        self.__Y = Y
        """Coordinates of the customer points""" 
        
        self.__w = w
        """Weights of the customer points""" 
        
        self.__prices = None
        """Dual variables of the covering constraints and
        the dual variable of the cardinality constraint"""
        
        self.__model = None 
        """MIP model of the pricing problem"""
        
        self.__X = None 
        """Coordinates of the facility in a solution"""
        
        self.__z = None 
        """Binary variables for selecting the customer subset"""
        
        self.__objf = None 
        """The model's objective function (as a constraint)"""
        
        self.__zero = 1.0E-05
                
        if use_mip: self.__init_price()
        
    #---------------------------------------------

    def __init_price(self):
        """
        Set up the pricing subproblem as a quadratically
        constrained MIP.
        """
        Y, w = self.__Y, self.__w  
        dmax = lambda i : max(euclid(Y[i],y) for y in Y)
        m = Y.shape[0]
    
        mod = Model('Pricing subproblem')
    
        X = mod.addVars( 2, lb=-mod.infinity ) 
        z = mod.addVars( m, vtype='B')
    
        # Continuous variables for the distance to the facility
        d = mod.addVars( m )
    
        # Continuous variables for the difference of the coordinates
        dx = mod.addVars( m, lb=-mod.infinity ) # X(0)-Y(i,0)
        dy = mod.addVars( m, lb=-mod.infinity ) # X(1)-Y(i,1)
    
        # Continuous variable equaling ||X-Y(i)||*z(i)
        delta = mod.addVars( m )

        # Constraints: X(0) - Y(i,0)==dx(i) and X(1) - Y(i,1)==dy(i)
        mod.addConstraints( X[0]-Y[i,0] == dx[i] for i in range(m) )  
        mod.addConstraints( X[1]-Y[i,1] == dy[i] for i in range(m) )  
    
        # Quadratic constraints to model the distance to the facility    
        mod.addQuadConstrs( dx[i]**2 + dy[i]**2 <= d[i]**2 for i in range(m) )
    
        # Constraints to ensure delta(i)=d(i) if z(i)=1
        mod.addConstraints( d[i] - dmax(i)*(1-z[i]) <= delta[i] for i in range(m) )
    
        # Objective function is \sum_i w(i)*delta(i) - lambda(i)*z(i).
        # As we, the time being, do not know lambda, we set each lambda(i)
        # temporarily to 1. Later, we have to change the objective coefficients
        # of variables z(i). As this cannot be done with GuRoBi, we include
        # the objective via an additional variable Obj and the constraint
        # Obj >= w*delta - lambda*z.
        Obj  = mod.addVar( lb = -mod.infinity )
        objf = mod.addConstraint( mod.sum(w[i]*delta[i] for i in range(m)) - mod.sum(z.values()) <= Obj,\
                                  return_constr=True )        
        mod.minimize( Obj )
    
        mod.update()
        mod.log_output = False 
    
        self.__model, self.__X, self.__z, self.__objf = mod, X, z, objf
        
    #---------------------------------------------
        
    def __miqcp_price_out(self):
        """
        Solve the pricing problem as as MIQCP and
        return the reduced cost and the 'column'
        showing smallest reduced cost.
        """
        M, X, z, prices  = self.__model, self.__X, self.__z, self.__prices    
        mu = -prices[-1] # dual variable cardinality constraint
    
        # Adjust objective coefficients of the z-variables
        coeffs = zip( z.values(), -prices[:-1])
        M.chgCoeffs( self.__objf, coeffs )
     
        # Let the MIP solver solve the pricing problem
        if not M.optim( ):
            print("Error: No solution to the pricing problem")
            return 0.0, None
    
        # Get smallest reduced cost
        best_bnd = M.ObjBound + mu
    
        # Extract the selected subset from solution to pricing problem
        column = dict()
        customers = list(M.get_solution(z, keep_zeros=False, precision=0.5 ).keys())
        column['subset'] = set( customers )
        column['X'] = np.fromiter( (M.get_solution(X).values()), dtype=float )
        column['cost'] = M.ObjVal + np.sum( prices[customers] )
        
        return best_bnd, column
        
    #---------------------------------------------
        
    def __drezner_price_out( self ):
        """
        Solve the pricing problem by solving a Weber problem with 
        limited distances using Drezner's algorithm. 
        """
        prices, Y, w = self.__prices, self.__Y, self.__w 
        # Remove customer points having zero prices
        custs = np.where( prices > self.__zero)[0]
        lobj, X = limitedDist( Y[custs], w[custs], prices[custs]/w[custs], screen='off' )
        redcst = lobj - prices.sum() # Note that \mu = -prices[-1]
        # Subset of customers is all i of weighted distance not exceeding the price
        customers = set( filter( lambda i : w[i]*euclid(X, Y[i]) < prices[i], range(len(Y)) ) )
        cost = sum( w[i]*euclid(X, Y[i]) for i in customers )
        column = dict()
        column['subset'] = customers
        column['X'] = X
        column['cost'] = cost 
        
        return redcst, column
    
    #---------------------------------------------
        
    def price_out(self):
        """
        Solves the pricing problem and returns the smallest
        reduced cost and corresponding column"""
        if self.__model is None: return self.__drezner_price_out()
        return self.__miqcp_price_out()
    
    #---------------------------------------------
    
    def end(self):
        """
        Free up space associated with the MIP pricing model
        """
        if not self.__model is None: self.__model.end()
        
    #---------------------------------------------
    
    @property
    def prices(self):
        """
        Return the current pricing vector/array.
        """ 
        return self.__prices 
    
    @prices.setter 
    def prices(self, value ):
        """
        Set the price vector/array to the array value.
        """
        self.__prices = value

#------------------------------------------------------------------------------

class Master:
    """
    Class for handling the master problem
    """
    def __init__(self, p, Y, w, screen='on' ):
        """
        Initialize the class. Obtain in particular an initial set
        of columns and set up the model of the master problem.
        
        Parameters
        ----------
        p : int  
            Number of facilities to locate
        Y : mx2 numpy array of float
            Coordinates of the customer points
        w : numpy array of float or int 
            The customer weights.
        screen : str 
            If 'on, log output when solving the pricing problem
            is send to stdout. Otherwise, let screen='off'.
        """
        self.__Y = Y 
        """Coordinates of the customer points"""
        
        self.__w = w 
        """Weights of the customer points"""
                  
        self.__y = None  
        """The master problem's decision variables"""
        
        self.__p = p 
        """Number of facilities to locate"""
        
        self.__constr= None 
        """numpy array of the master problem's constraints. 
        Constraints 0 to m-1 are the covering constraints,
        the last constraint is the cardinality contraint."""
        
        self.__intTol = 1.0E-04
        """Tolerance for variables to be integer"""
        
        self.__itr = 0 
        """Number of times a restricted linear master is solved"""
        
        self.__talk = screen == 'on'
        """Display some logout to stdout if screen is on"""
        
        self.__columns = None 
        """Set of columns (customer subsets) generated"""
        
        self.__model = None 
        """The MIP/LP model of the master problem"""
        
        # Obtain set of initial columns from heuristic solutions
        self.__get_initial_columns( )
        
        # Initialize the master problem
        self.__init_master()
        
    #----------------------------------------------

    def __add_subsets( self, X, a ):
        """
        Add the columns corresponding to the solution (X,a) to 
        the current list of columns. Thereby X is the coordinates 
        of the p facilities and a the customer assignment to these 
        p facilities.
        """
        Y, w = self.__Y, self.__w 
        ass = np.array(a)
        for j in range( self.__p ):
            j_set = set( np.where( ass==j )[0] )
            # Check if subset already in the list
            in_list = False
            for c in self.__columns:
                a_set = c['subset']
                in_list = a_set == j_set
                if in_list: break
            if not in_list:
                new_col = dict()
                new_col['subset'] = j_set
                new_col['X'] = X[j] 
                new_col['cost'] = sum( euclid(X[j],Y[i])*w[i] for i in j_set )
                self.__columns.append(new_col)
        
    #----------------------------------------------

    def __get_initial_columns( self ):
        """
        Obtain a collection of initial columns (= customer subsets)
        from heuristic solutions.
        """
        self.__columns, m = list(), len(self.__Y)
        if self.__talk:
            print('Finding initial columns using location-allocation {0:d} times'.format(m))
        for _ in range(m):
            _, Xsol, a, _ = locAlloc(self.__p, self.__Y, self.__w )
            self.__add_subsets( Xsol, a )
        
        if self.__talk: 
            print("Applying p-median heuristic for finding further columns.")
            screen = 'on' if self.__talk else 'off'
        _, Xsol, a, _ = pmedian_heuristic( self.__p, self.__Y, self.__w, screen=screen )
        self.__add_subsets( Xsol, a ) 

    #----------------------------------------------

    def __init_master( self ):
        """
        Set up the initial master problem using the initial columns. Model
        thereby the problem as a set covering problem. Return then
        the model. Later we need to access the objective function, which
        can be done using model.getObjective(). As there is only one
        type of variables, we can also access them via model functions
        model.getVars(). Similar, we can obtain all (linear) constraints
        using model.getConstraints(). Note that the first m constraints
        are covering constraints. The last constraints limits the number
        of customer subsets to be no larger than p.
        """
        cols = self.__columns
        ncols = len(cols)
        M = Model('Master problem')
        self.__y = M.addVars( ncols )
        M.minimize( M.sum( cols[j]['cost']*self.__y[j] for j in range(ncols) ) )
   
        # Include the covering constraints for each customer
        m  = len(self.__Y)
        row = lambda i : filter( lambda j : i in cols[j]['subset'], range(ncols) )
        cnstr = M.addConstraints( (M.sum( self.__y[j] for j in row(i) ) >= 1 \
                                  for i in range(m)), return_constr=True )

        # Add constraint on number of facilities. Note that the
        # dual variable to a <=-constraint should be non-positive #
        # when minimizing the objective function,
        cnstr.append( M.addConstraint(M.sum(self.__y.values()) <= self.__p, return_constr=True) )
        
        M.update()
        M.log_output = False 
        self.__constr = np.array( cnstr )
        self.__model = M 

    #----------------------------------------------

    def add_column ( self, column  ):
        """
        Add a generated column (i.e. customer subset) 
        as a column to the master problem.
        """
        rows = list(column['subset']) # covering constraints for the set
        rows.append( len(self.__Y) )
        new_y = self.__model.addVar(obj=column['cost'], constr=self.__constr[rows], \
                                  coeffs=1.0 )  
        ncols = len(self.__y)
        self.__y[ncols] = new_y
             
    #----------------------------------------------

    def __get_sol( self ):
        """
        Transform the solution to the integer master program to
        a solution for the multi-source Weber problem.
        """
        objv = self.__model.ObjVal
        selected = self.__model.get_solution( self.__y, keep_zeros=False, precision=0.5 ).keys()
        Xsol = np.array( [ self.__columns[j]['X'] for j in selected] )
        a = np.zeros( len(self.__Y), dtype=int )
        for num, j in enumerate(selected):
            cust_lst = list(self.__columns[j]['subset'])
            a[cust_lst] = num
    
        return objv, Xsol, a   
        
    #----------------------------------------------

    def __master_integer( self ):
        """
        Checks if last solution to the linear master problem is integer
        """
        one = 1.0 - self.__intTol
        fract = np.any( (np.fromiter( self.__model.get_solution(self.__y, keep_zeros=False,\
                             precision=self.__intTol).values(), dtype=float ) ) < one )
        return not fract

    #----------------------------------------------
    
    def solve(self):
        """
        Solve the current LP master program. Return
        the master program's objective value and the
        dual values.
        """
        self.__itr += 1

        if self.__model.optim( ):
            prices = np.array(self.__model.get_dual_values(self.__constr))
            return self.__model.ObjVal, prices
        return None, None     
         
    #----------------------------------------------
    
    def solve_as_IP( self ):
        """
        Solve the final master as an integer program.
        """
        if self.__master_integer(): return self.__get_sol()
        
        self.__model.log_output = self.__talk 
        for y in self.__y.values():
            y.lb, y.ub = 0, 1
            self.__model.set_vartype( y, 'B' )
        if self.__talk: print("Solving the integer master program")
        if self.__model.optim( ): return self.__get_sol(  )
        if self.__talk: print("Merde alors! No integer master solution")
        return 0.0, None, None
    
    #----------------------------------------------
    
    def end(self):
        """
        Free the LP/MIP model
        """
        if not self.__model is None: self.__model.end()
        
    #----------------------------------------------
    
    @property 
    def ncols(self):
        """
        Number of columns currently in the master
        """
        return len(self.__y)
    
    @property
    def itr(self):
        """
        Number of times master problem has been solved.
        """
        return self.__itr 
        
#------------------------------------------------------------------

def set_optTol( value = None ):
    """
    Set/return the relative optimality tolerance to be applied.
    """
    global __optTol
    
    if value is None: return __optTol
    if 0.0 < value < 1.0: __optTol = value

#------------------------------------------------------------------

def colgen( p, Y, w, mip_pricing=True, max_iter=float('inf'), screen='off' ):
    """
    Solve multi-source Weber problem by means of column generation.
        
    Parameters
    ----------
    p : intLB, objv, Xsol, a, itr
        number of facilities to locate (1 < p < #customers) 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or numpy array of m float or int
        positive customer weights (if None, all weights are set to 1)
    mip_pricing : bool
        If True, the pricing problem is solved using as a MIQCP using
        the MIP solver. Otherwise, it is solved as a Weber problem with 
        limited distances by means of Drezner's algorithm.
    max_iter : int or float
        Maximal number of column generation iterations (solving a
        restricted linear master problem) to be applied.
    screen: 
        If 'on', intermediate output on the state of the computations 
        is printed to stdout.

    Returns
    -------
    objv : float
        objective value of the best solution found
    lbnd : float
        lower bound from the LP relaxation of the set-covering problem
    Xsol : px2 numpy array of float
        the p facility's coordinates
    a : list of int
        the assignment of customer points to the facilities,
        i.e., a[i]=j if point Y[i] is assigned to facility at 
        location X[j]
    itr : int
        number of iterations performed
    """
    talk = screen=='on'

    if w is None: w = np.ones( len(Y), dtype=int )
    
    if talk:
            print('-'*72) 
            print("Column generation for multi-source Weber -- Initializing")
            if mip_pricing:
                print("    Using a MIQCP to price out columns.")
            else:
                print("    Using Drezner's algorithm to price out columns.")

    # Create an instance of the pricing problem class
    price_prob = Pricing( Y, w, use_mip=mip_pricing )
   
    # Intialize the master problem
    master = Master( p, Y, w, screen=screen ) 
    
    # Do the actual column generation
    if talk: 
        print()
        print('-'*72)
        print("Column generation for multi-source Weber -- Iterating")
        print("-"*72)
        print("Iter   columns   current_LB   lower_bound   upper_bound   reduced_cost")
        print("-"*72)
            
    
    LB = 0.0 # lower bound on the linear master problem's objective
    while True:
        # Solve master problem
        UB, price_prob.prices = master.solve()
        if UB is None:
            print("Warning: Failed to solve master problem")
            break
        # Solve pricing problem
        rc_lb, column = price_prob.price_out()
        # Adjust the lower bound
        cur_LB = UB + p*rc_lb 
        if cur_LB > LB: LB = cur_LB 
        if (UB - LB)/max(1.0,LB) < __optTol: break 
        if master.itr >= max_iter: break
        # Add generated column to master problem
        master.add_column(column)
        # Display state of the computations   
        if talk:
            print('{0:4d}   {1:7d}   {2:10.2f}   {3:11.2f}   {4:11.2f}   {5:12.2f}'.format(\
                master.itr, master.ncols, cur_LB, LB,UB, rc_lb) )
             
    # Release the model for the pricing problem. No more needed
    price_prob.end()
    
    # Solve the master as an integer program 
    objv, Xsol, a = master.solve_as_IP()
    
    # Release master IP model
    master.end()
    
    return LB, objv, Xsol, a, master.itr
