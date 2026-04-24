"""
    Module plane.mip of package pyloa:
    2nd order cone (MIP) models of planar location problems solved
    by means of Cplex's or GuRoBi's solver.
"""
import numpy as np
from pyloa.mip.model import Model
from itertools import product 
from pyloa.util import euclid

#------------------------------------------------------------------

def SOCPweber( Y, w=None, screen='off' ):
    """
    Solves the Fermat-Weber problem as a 2nd order cone problem
    using Cplex's solver.
    
    Parameters
    ---------- 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or numpy array of m float or int
        positive customer weights (if None all weights are set to 1)
    screen : str 
        If 'on', intermediate output on the state of
        the computations are printed to the screen.
 
    Returns
    -------
    objv : float
        Sum of weighted distances to the optimal facility location 
    X   : px2 numpy array of float
        Coordinates of the optimal facility location
    """
    n = Y.shape[0]
        
    M = Model('Fermat-Weber')
    
    # Locations of the p facilities
    x = M.addVars( 2, lb = -M.infinity )
      
    # Distance to facility site
    d = M.addVars( n )
    
    # Variables equaling x-Y[i]
    v = M.addVarMatrix( n, 2, lb=-M.infinity )
    
    # Constraints v[i] = x-Y[i]
    M.addConstraints( x[dim]-Y[i,dim] == v[i,dim] for dim in range(2) for i in range(n) )
    
    # Quadratic constraints modelling the distance to the facility  
    M.addQuadConstrs( v[i,0]**2 + v[i,1]**2 <= d[i]**2 for i in range(n) ) 
        
    # Objective function: Minimize sum of weighted distances
    if w is None:
        M.minimize( M.sum( d ) )
    else:
        M.minimize( M.sum( w[i]*d[i] for i in range(n) ) )
   
    # Call the solver
    objv, X = 0.0, None  
    M.log_output = screen=='on'
    ok = M.optim( )
    if ok:
        objv = M.ObjVal 
        X = np.fromiter( (M.get_solution(x).values()), dtype=float )
    M.end()
     
    return objv, X

#------------------------------------------------------------------

def SOCPcenter( Y, w, screen='off'):
    """
    Solve the weighted 1-center problem as a quadratically 
    constrained continuous convex optimization problem using 
    a MIP solver. For reasons of numerical stability, we normalize 
    the weights w by dividing it by the average weight.
    
    Parameters
    ----------
    Y : numpy mx2 array of float
        m customer points in Euclidian plane
    w : numpy array of m int or float
        positive customer weights
    screen: str
        if 'on', information on solution process is printed
        
    Returns
    -------
    r : float
        "radius" of the computed center location
    X : numpy array of two float
        coordinates of the center as a 
    """
    n = Y.shape[0]
    PCP = Model('Planar 1-center problem')
    
    # Locations of the p facilities
    
    X = PCP.addVars( 2, lb=-PCP.infinity )
    
    # Variables equaling X-Y[i]
    V = PCP.addVarMatrix( n, 2, lb=-PCP.infinity )
    
    # objective value (that is squared "radius") of the solution
    r2 = PCP.addVar()
    
    # Constraints: Constraints V[i] = X-Y[i]
    PCP.addConstraints( X[dim]-Y[i,dim] == V[i,dim] for dim in range(2) for i in range(n) )
    
    # Quadratic constraints: radius not smaller than weighted distances to X  
    w_mean = 1.0
    if (w is None) or np.all(w==1):  
        PCP.addQuadConstrs( V[i,0]**2 + V[i,1]**2 <= r2 for i in range(n) ) 
    else:
        w_mean = w.sum()/len(w) 
        ww = (w/w_mean)**2
        PCP.addQuadConstrs( ww[i]*(V[i,0]**2 + V[i,1]**2) <= r2 for i in range(n) )
        
    PCP.minimize( r2 )
    
    # Call the solver
    PCP.log_output = screen=='on'
    ok = PCP.optim( )
    if ok:     
        r = np.sqrt(PCP.ObjVal)*w_mean
        XX = np.fromiter( (PCP.get_solution(X).values()), dtype=float )
        
    PCP.end() 
    if ok: return r, XX    
    return 0.0, None
    
#------------------------------------------------------------------

def weber_mipq( p, Y, w=None, screen='off', strategy=0, timLim=None ):
    """
    Formulates the multi-source Weber problem as a quadratically
    constrained MIP and solves it using a MIP solver.
    
    Parameters
    ----------
    p : int
        number of facilities to locate (1 < p < #customers) 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or numpy array of m float or int
        positive customer weights (if None all weights are set to 1)
    screen: 
        If 'on', intermediate output on the state of
        the computations are printed to the screen.
    strategy : int 
        strategy Cplex or GuRoBi applies to solve the the problem:
           
        * 1 -> solve the quadratic continuous relaxation
        * 2 -> use conical cuts (or outer approximation) 
        * 0 -> decide automatically  
    timeLim : float 
        Time limit (seconds) to be used for the MIQCP solver

    Returns
    -------
    objv : float  
    
    Xsol : px2 numpy array of float  
    
    a : list of int
        the assignment of customer points to the facilities,
        i.e., a[i]=j if point Y[i] is assigned to facility at 
        location X[j]
    
    tim : float
        Computation time in seconds
    
    nnodes : int
        number of B&B tree nodes
    """
    # Function returning a customer's distance to most distant customer
    dmax = lambda i : max( euclid(Y[i],y) for y in Y )
    m = Y.shape[0]
        
    MWP = Model('Multi-source Weber')
    
    # Locations of the p facilities
    X = MWP.addVarMatrix( p, 2, lb=-MWP.infinity )
    
    # Assignment variables
    z = MWP.addVarMatrix( m, p, vtype='B') 
    
    # Distance to closest facility site
    d = MWP.addVars(m)
    
    # Variables equalling X[j]-Y[i]
    V = MWP.addVarCube((m,p,2),lb=-MWP.infinity)
    
    # Distance between customers and facilities
    dmat = MWP.addVarMatrix( m, p )
    
    # Constraints: Each customer need to be assigned to a facility
    MWP.addConstraints( MWP.sum( z[i,j] for j in range(p) ) == 1 for i in range(m) )
    
    # Constraints: Constraints to ensure that d[i] = dmat[i,j] if z[i,j]=1
    MWP.addConstraints( dmax(i)*(1-z[i,j])+d[i] >= dmat[i,j] for i,j in product(range(m),range(p)) )
    
    # Constraints: Constraints V[i,j] = X[j]-Y[i]
    MWP.addConstraints( X[j,dim]-Y[i,dim] == V[i,j,dim] for i,j,dim in product(range(m),range(p),range(2)) )

    # Constraints: Quadratic constraints to model the Euclidian distances
    MWP.addQuadConstrs( V[i,j,0]**2 + V[i,j,1]**2 <= dmat[i,j]**2 for i,j in product(range(m),range(p)) ) 

    # Constraints: For purposes of symmetry breaking order X from west to east
    MWP.addConstraints( X[j-1,0] <= X[j,0] for j in range(1,p) )
    
    # Objective function
    if w is None: 
        MWP.minimize( MWP.sum(d) )
    else:
        MWP.minimize( MWP.sum( w[i]*d[i] for i in range(m) ) )
    
    # Choose strategy for solving the linear relaxation
    MWP.MIQCPMethod = strategy
    
    # Set the time limit if one is given
    if not timLim is None: MWP.timelimit = timLim
    
    # Call the solver
    MWP.log_output = not screen=='off'
    ok = MWP.optim( )
    
    # Return None if solving the model failed
    if not ok:
        MWP.end()
        return 0.0, None, None, 0.0
        
    objv  = MWP.ObjVal
    XX    = MWP.get_solution(X)
    Xsol  = np.zeros( (p,2) )
    for i,j in XX: Xsol[i,j] = XX[(i,j)]
    
    assign = (MWP.get_solution(z, keep_zeros=False, precision=0.1)).keys()
    a = list( pair[1] for pair in assign )
    
    tim = MWP.runtime 
    nnodes = MWP.nodeCount
    MWP.end()
    
    return objv, Xsol, a, tim, nnodes

#------------------------------------------------------------------

def pcenter_mipq( p, Y, w, screen='off', strategy=0, timeLim=None ):
    """
    Formulates the planar p-center problem as a quadratically
    constrained MIP and solves it using a MIP solver.
    
    Parameters
    ----------
    p : int
        number of facilities to locate (1 < p < #customers) 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : numpy array of m float or int
        positive customer weights
    screen: str
        If 'on', the MIP solver shows intermediate output on the state of
        the computations are printed to the screen.
    strategy : int 
        Strategy the MIP solver should apply to solve the problem:
        
        * 0 -> decide automatically 
        * 1 -> solve the quadratic continuous relaxtion for obtaining lower bounds
        * 2 -> use conical cuts
    timeLim : float 
        Time limit (seconds) to be used for the MIQCP solver

    Returns
    -------
    objv : float
        objective value of the optimal solution (smallest
        weighted distance to the nearest center point)
    Xsol : px2 numpy array of float
        Xsol[j] gives the two coordinates of the j-th center point
    a : list of int
        the assignment of customer points to the centers,
        i.e., a[i]=j if point Y[i] is assigned to the center at 
        location X[j]
    tim : float
        Computation time in seconds
    nnodes : int
        number of B&B tree nodes
    """
    # Function returning a customer's distance to most distant customer
    dmax = lambda i : max( euclid(Y[i],y) for y in Y )
    m = Y.shape[0]
        
    PPCP = Model('Planar p-center problem')
    
    # Locations of the p facilities
    X = PPCP.addVarMatrix( p, 2,lb=-PPCP.infinity )
    
    # Assignment variables
    z = PPCP.addVarMatrix( m, p, vtype='B' ) 
    
    # "Radius" (largest weighted distance to nearest center point)
    r = PPCP.addVar( )
    
    # Variables equaling X[j]-Y[i]
    V = PPCP.addVarCube( (m,p,2), lb=-PPCP.infinity )
    
    # Distance between customers and facilities
    d = PPCP.addVarMatrix( m, p )
    
    # Constraints: Each customer need to be assigned to a center
    PPCP.addConstraints( ( PPCP.sum( z[i,j] for j in range(p) ) == 1 ) for i in range(m) )
    
    # Constraints: Constraints to ensure that the radius is at least the weighted
    # distance if if z[i,j]=1
    if w is None:
        PPCP.addConstraints( (d[i,j]-dmax(i)*(1-z[i,j]) <= r for i in range(m) for j in range(p)) )
    else:
        PPCP.addConstraints( (w[i]*(d[i,j]-dmax(i)*(1-z[i,j])) <= r for i in range(m) for j in range(p)) )
        
    # Constraints: Constraints V[i,j] = X[j]-Y[i]
    PPCP.addConstraints( ( X[j,dim]-Y[i,dim] == V[i,j,dim] for dim in range(2) for j in range(p) \
                       for i in range(m) ) )
    
    # Constraints: Quadratic constraints to model the Euclidian distances
    PPCP.addQuadConstrs( ( V[i,j,0]**2 + V[i,j,1]**2 <= d[i,j]**2 \
                           for i in range(m) for j in range(p) ) )

    # Constraints: For purposes of symmetry breaking order X from west to east
    PPCP.addConstraints( ( X[j-1,0] <= X[j,0] for j in range(1,p) ) )
    
    # Objective function
    PPCP.minimize( r )
    
    # Choose strategy for solving the linear relaxation
    PPCP.MIQCPMethod = strategy 
    
    # Set time limit if one is given
    if not timeLim is None: PPCP.timelimit = timeLim
        
    # Call the solver
    PPCP.log_output = not screen=='off'
    ok = PPCP.optim( )
    
    # Return None if solving the model failed
    if not ok:
        PPCP.end()
        return 0.0, None, None, 0.0
    
    objv  = PPCP.ObjVal
    XX    = PPCP.get_solution(X)
    Xsol  = np.zeros( (p,2) )
    for i,j in XX: Xsol[i,j] = XX[(i,j)]
    
    assign = (PPCP.get_solution( z, keep_zeros=False, precision=0.1 )).keys()
    a = list( pair[1] for pair in assign )
    
    tim = PPCP.runtime
    nnodes = PPCP.nodeCount
    PPCP.end()
    
    return objv, Xsol, a, tim, nnodes
