"""
    Module plane.loc_alloc of package pyloa:
    Methods for solving the multi-source Weber and planar p-center
    problem mostly relying on location-allocation.
"""
import numpy as np

from pyloa.util import euclid, all_two_parts
from pyloa.plane.weber import solveWeber
from pyloa.plane.center import charalambous

from sklearn.cluster import AgglomerativeClustering as hAggClust
from sklearn.cluster import KMeans

from pyloa.mip.model import Model
from itertools import product

#------------------------------------------------------------------

def _lex_smaller( x, y ):
    """
    Returns True if array x is lexicograpically smaller than y.
    """
    if x.shape != y.shape: return False 
    if np.all( x>=y ): return False
    i = np.where( x != y )[0][0] 
    return x[i] < y[i]

#------------------------------------------------------------------

def _locAlloc( p, Y, w, X0=None, minisum=True, screen='off' ):
    """
    Cooper's location-allocation heuristic.

    Parameters
    ----------
    p : int
        number of facilities to locate (1 < p < #customers)
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : numpy array of m float or int
        positive customer weights
    X0 : None or a px2 numpy array of float
         Optional initial solution. If an initial solution is
         passed it need to be a numpy px2 array.
    minisum : bool 
        True, if a multi-source Weber problem has to be solved
        and False if a planar p-center problem is to be solved.
    screen: string
        If 'on', the result of the current iteration is printed 
        to "stdout"

    Returns
    -------
    totalCost : float
        the objective value of the solution
    X : px2 numpy array of float
        the solution's p facility locations
    itr : int
        the number of location-allocation iterations
    """
    locFunc = solveWeber if minisum else charalambous
    m = Y.shape[0]
    if X0 is None:
        X = np.zeros((p,2))
        #Select p customer points randomly as initial locations
        sel = np.random.choice( m, p, replace=False )
        X = Y[sel]
    else: 
        X = X0 
        p = X.shape[0]
        
    a = [0]*m
    objv = np.Inf if minisum else np.full(p,np.Inf)
    costj = np.zeros(p,dtype=float)
    itr = 0
    epsilon = 1.0E-04
    screenOn = screen.lower() == 'on'
    if screenOn:
        print("------ LOCATION-ALLOCATION HEURISTIC -----")
        print("Iter  Current cost")
    
    while True:  
        itr += 1
        # Allocation: Assign customers to the closest location
        a = [np.argmin(np.fromiter((euclid(X[j],y) for j in range(p)),float)) for y in Y]
        # Location: Solve the single Weber problem for each region 
        for j in range(p):
            # Solve single Weber problem over j-th customer subset
            customers = list( filter( lambda i : a[i]==j, range(m) ) )
            costj[j], X[j] = locFunc( Y[customers], w[customers] )
        # Print cost of current solution
        if minisum: 
            cur_obj = costj.sum()
            if screenOn: print( '{0:4d}  {1:12.4f}'.format(itr,cur_obj))
            if cur_obj > objv - epsilon: break 
            objv = cur_obj     
        else:
            costj[:] = -np.sort( -costj )
            if screenOn: print( '{0:4d}  {1:12.4f}'.format(itr,costj[0]))
            if not _lex_smaller(costj, objv ): break
            objv[:] = costj[:]    
        
    if screenOn: print("------------------------------------------")

    val = objv if minisum else objv[0]
    return val, X, itr                 

#------------------------------------------------------------------

def locAlloc( p, Y, w=None, minisum=True, initLA='random', repeat=1,\
              screen='off' ):
    """
    Applies Cooper's location allocation heuristic using different
    initial solutions and returns the best one found.
    
    Parameters
    ----------
    p : int
        number of facilities to locate (1 < p < #customers) 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or a numpy array of m float or int
        positive customer weights (if None weights will be set to 1)
    minisum : bool
        If True, a multi-source Weber problem is solved. Otherwise,
        a planar p-center problem.
    initLA : string
        method to be applied for finding initial solution.
        Currently the following two options can be chosen.
        (i) 'random' means a random initial solution at the 
        customer points
        (ii) 'cluster' means initial solution from clustering
        using the kmeans as well as hierarchical methods 
        based on complete, single linkage and Ward's method
    repeat : int
        number of times location-allocation is repeated with
        different start solution (only applies if initLA='random')
    screen: string
        If 'on', the result of the current iteration is printed 
        to "stdout"

    Returns
    -------
    bestObj : float
        the objective value (as a float) of the solution
    Xbest : px2 numpy array of float
        the solution's p facility locations,
    a : list of int
        the assignment of customer points to the facilities,
        i.e., a[i]=j if point Y[i] is assigned to facility at 
        location X[j]
    itr : int
        the total number of location-allocation iterations.
    """
    locFunc = solveWeber if minisum else charalambous
    screen = screen.lower()
    screen_on = screen == 'on'
    initLA = initLA.lower()
    X = np.zeros((p,2))
    Xbest = np.zeros((p,2))
    bestObj = np.Inf
    tot_itr = 0
    if w is None : w = np.ones(len(Y),dtype=int)
    
    if initLA == 'cluster':
        for link in ['average','complete','single','ward','kmeans']:
            if screen_on: print("Cluster customer locations using =",link)
            # Apply clustering method and return array of cluster labels
            myCluster = KMeans(n_clusters=p) if link=='kmeans' else\
                        hAggClust(n_clusters=p, linkage=link)
            clusters = myCluster.fit_predict(Y)
            # Locate a facility in each cluster
            for j in range(p):
                customers = clusters==j
                _, X[j] = locFunc(Y[customers], w[customers])
            # Improve solution using location-allocation
            objVal, X, itr = _locAlloc( p, Y, w, X, minisum=minisum, screen=screen )
            tot_itr += itr
            if objVal < bestObj:    
                bestObj = objVal
                Xbest[:,:] = X                  
    else:
        for trial in range(repeat):
            if screen_on: print("Trial no.",trial+1,"with random start")
            objVal, X, itr = _locAlloc(p, Y, w, minisum=minisum, screen=screen)
            tot_itr += itr
            if objVal < bestObj:
                bestObj = objVal
                Xbest[:,:] = X
                
    # Find customer assignment in best solution found above
    a = [np.argmin(np.fromiter((euclid(Xbest[j],y) for j in range(p)),float)) for y in Y]

    # Return the best solution
    return bestObj, Xbest, a, tot_itr

#------------------------------------------------------------------

def pmedian_heuristic( p, Y, w=None, minisum=True, screen='off' ):
    """
    p-median heuristic for the multi-source Weber problem. We
    first solve a p-median problem using a MIP solver in order to cluster
    the customer set. The optimal locations for each cluster are
    then used as starting point for the location-allocation method.

    Parameters
    ----------
    p : int
        number of facilities to locate (1 < p < #customers) 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or a numpy array of m float or int
        positive customer weights (if None all weights are set to 1)
    minisum : bool
        If True, a multi-source Weber problem is solved; otherwise
        a planar p-center problem
    screen: 
        If 'on', Cplex's intermediate output on the state of
        the computations are printed to the screen.

    Returns
    -------
    objv : float  
    
    Xsol : px2 numpy array of float    
    
    a : list of int  
        the assignment of customer points to the facilities,
        i.e., a[i]=j if point Y[i] is assigned to facility at 
        location X[j]    
    """
    pmed = Model('p-median')
    m = Y.shape[0]
    if w is None: w = np.ones( m, dtype=int )
    
    # Assignment variables (continuous, as they are integer in optimum)
    z = pmed.addVarMatrix( m, m )
    
    # Binary variables equal to 1 if point i represents a cluster
    y = pmed.addVars(m, vtype='B') 
    
    # Assignment constraints   
    pmed.addConstraints( pmed.sum(z[i,j] for j in range(m)) == 1 for i in range(m) ) 
                           
    # No assignment to an inactive possible representative
    pmed.addConstraints( z[i,j]-y[j] <= 0 for i,j in product(range(m),range(m)) )
    
    # Number of clusters
    pmed.addConstraint( pmed.sum(y) == p )
    
    # Objective: minimize sum of distances to representative
    pmed.minimize( pmed.sum(euclid(Y[i],Y[j])*z[i,j] for i,j in product(range(m),range(m)) ) )
                             
    # Solve model
    pmed.log_output = screen=='on'
    ok = pmed.optim( )
    if not ok:
        pmed.end()
        return 0.0, None, None
      
    # Retrieve solution to p-median
    ysol = pmed.get_solution( y, keep_zeros=False, precision=0.1 ).keys() 
     
    pairs = pmed.get_solution( z, keep_zeros=False, precision=0.1 ).keys()
    assign = np.array( [pair[1] for pair in pairs] ) # Assignment of customers to facilities
    
    # Solve Fermat-Weber for each cluster
    locFunc = solveWeber if minisum else charalambous
    X = np.zeros( (p,2) )
    cnt = 0
    for j in ysol:
        customers = assign == j 
        _, X[cnt] = locFunc(Y[customers], w[customers])
        cnt += 1
                
    # Improve solution using location-allocation
    objv, X, itr= _locAlloc( p, Y, w, X, minisum=minisum, screen=screen )
    a = [np.argmin(np.fromiter((euclid(X[j],y) for j in range(p)),float)) for y in Y]
    
    pmed.end()
    return objv, X, a, itr
     
#------------------------------------------------------------------

def weber_vns( p, Y, w=None, X=None, minisum=True, screen='off' ):
    """
    Variable neighbourhood search for respectively the multi-source 
    Weber and planar p-center problem in the realm of the VNS suggested 
    by Brimberg et al. (2000). Let k be the neighbourhood size. In each 
    iteration, the method tries to find an improved solution by picking 
    k customer points and locating facilities on these k customer points. 
    We choose the k customers with a probability proportional to the 
    weighted distance to their nearest facility. In exchange of locating 
    a facility at a selected customer point, the currently open facility 
    closest to that customer is closed.
    
    Reference: Brimberg et al. (2000). Improvements and comparison of 
    heuristics for solving the multi-source Weber problem. Operations 
    Research 48: 444-460.
    
    Parameters
    ----------
    p : int
        number of facilities to locate (1 < p < #customers) 
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or numpy array of m float or int
        positive customer weights (if None all weights are set to 1)
    X : None or px2 numpy array of float
        If not none, X is the initial solution with X[j] specifying
        the two coordinates of the j-th facility, j=0,...,p-1.
    minisum : bool
        If True, a multi-source Weber problem is solved; otherwise
        a planar p-center problem.
    screen : str 
        If 'on', Cplex's intermediate output on the state of
        the computations are printed to the screen.

    Returns
    -------
    objv : float
    
    X : px2 numpy array of float
    
    a : list of int
        the assignment of customer points to the facilities,
        i.e., a[i]=j if point Y[i] is assigned to facility at 
        location X[j]
    """
    talk = screen == 'on'
    m = len(Y)
    if w is None: w = np.ones(m, dtype=int)
    # Find initial solution 
    if X is None: 
        if talk: 
            print("-"*50)
            print("Computing initial solution")
        objv, X, a, _ = locAlloc( p, Y, w, minisum=minisum, initLA='cluster')
        if talk: print("Initial objective value:",objv)
    else:
        a = [np.argmin(np.fromiter((euclid(X[j],y) for j in range(p)),float)) for y in Y]
        objv = sum( w[i]*euclid(X[a[i]],Y[i]) for i in range(m) ) if minisum else\
            max( w[i]*euclid(X[a[i]],Y[i]) for i in range(m) )
        
    # Smallest and largest neighbourhood size
    kmin = 1
    kmax = min(p,m-p)
    k_cur = 1     
    
    # Maximal number of subsequent non-improving iterations
    max_fail = min( max(10,m//2), 50 )
    num_fail = 0
    
    if talk:
        print("-"*50)
        print("Iter  Current_obj_val  Trial_obj_val  Best_obj_val")
        print("-"*50)
    # Do the iterations
    best_objv = objv 
    best_X = X.copy()
    best_a = a.copy()
    X_cur = X.copy()
    itr = 0
    while num_fail < max_fail:
        itr += 1
        X_cur[:] = X[:]
        prob = np.fromiter( (w[i]*euclid(Y[i],X[a[i]]) for i in range(m)), dtype=float )
        to_open = np.random.choice(range(m), k_cur, False, prob/prob.sum() )
        if k_cur < p:
            # Find k_cur current facility locations to be closed
            closed = [False]*p
            for i in to_open:
                k = a[i]
                if closed[k]: 
                    k = np.argmin(np.fromiter((euclid(X[j],Y[i]) for j in range(p) if not closed[j]),float))  
                closed[k] = True
            num = 0
            for j in range(p):
                if closed[j]:
                    X[j] = Y[to_open[num]]
                    num += 1    
                    if num == len(to_open): break
        else: 
            X[:] = Y[to_open]
        trial_objv, X, _ = _locAlloc( p, Y, w, X, minisum=minisum )
        if talk: 
            print('{0:4d}  {1:15.2f}  {2:13.2f}  {3:12.2f}'.format(itr,objv,trial_objv,best_objv))
        if trial_objv < objv:
            # Solution accepted as new current one 
            objv = trial_objv 
            a = [np.argmin(np.fromiter((euclid(X[j],y) for j in range(p)),float)) for y in Y]
        else: 
            # Reject solution as new current one
            X[:] = X_cur[:]
        # Check if new best solution found
        if objv < best_objv:
            best_objv = objv 
            best_X[:] = X[:]
            best_a[:] = a[:]
        else:
            num_fail += 1
            if num_fail == max_fail: break
            k_cur += 1
            if k_cur > kmax: k_cur = kmin
        
    if talk: print("-"*50)
    return best_objv, best_X, best_a  
    
#------------------------------------------------------------------

def twoFacility( Y, w, minisum=False, screen='off'):
    """
    Solves the two-facility Weber or 2-center problem by investigating all
    possible partitions of the customer set in two non-overlapping
    convex hulls.
    
    Parameters
    ----------
    Y : mx2 numpy array of float
        m customer points in Euclidian plane
    w : None or numpy array of m float or int
        positive customer weights (if None all weights are set to 1)
    minisum : bool 
        If True, a 2-Weber is solved; otherwise a 2-center problem
    screen : str 
        If 'on', intermediate output on the state of the computations 
        is printed to the screen.
    
    Returns 
    -------
    objv : float  
        The solution's objective value 
    X : 2x2 numpy array of float
        X[0] and X[1] are the coordinate vectors of the two facility
        locations
    a : list of int
        the assignment of customer points to the facilities,
        i.e., a[i]=j if point Y[i] is assigned to facility at 
        location X[j], j=0 or 1
    """
    talk = screen.lower()=='on'
    if talk:
        print("-"*50) 
        print('Creating all partitions in two non-overlapping convex hulls')
    
    all_partits = all_two_parts(Y)
    locFunc = solveWeber if minisum else charalambous
    
    if talk: 
        num = sum( len(all_partits[num]) for num in all_partits )
        print(num,'partitions generated')
        print("Partition no.  Objective value")
        print("-"*30)

    m = len(Y)
    X = np.zeros((2,2),dtype=float )
    if w is None: w = np.ones(m,dtype=int)
    best = np.infty if minisum else (np.infty,np.infty)
    X_best = np.zeros( (2,2),dtype=float )
    a = np.zeros(m, dtype=int )
    itr = 0
    for num in all_partits:
        for I1, I2 in all_partits[num]:
            itr += 1   
            # Solve Fermat-Weber problem for the two customer sets
            obj0, X[0] = locFunc( Y[I1], w[I1] )
            obj1, X[1] = locFunc( Y[I2], w[I2] )
            objv = obj0+obj1 if minisum else (max(obj0,obj1),min(obj0,obj1)) 
            if talk: 
                cur_obj = objv if minisum else objv[0]
                print('{0:13d} {1:16.2f}'.format(itr, cur_obj) )
            improve = objv < best if minisum else (objv[0] < best[0]) or\
                      ( (not objv[0] > best[0]) and (objv[1] < best[1]) )  
            if improve:
                best = objv 
                X_best[:,:] = X[:,:] 
                a[I1] = 0
                a[I2] = 1
    if talk: 
        print('-'*50)
        mybest = best if minisum else best[0]
        print('Optimal objective value: {0:<.2f}'.format(mybest))
        print('Facility no. 0 at      : ({0:<.4f},{1:<.4f})'.format(X[0,0],X[0,1]))
        print('Facility no. 1 at      : ({0:<.4f},{1:<.4f})'.format(X[1,0],X[1,1]))
        print('Assignment of customer points')
        print('Point   :',end='')
        for i in range(m): print(' {0:4d}'.format(i),end='')
        print()
        print('Facility:',end='')
        for f in a: print(' {0:4d}'.format(f),end='')
        print()
        print('-'*50)
                
    return best, X_best, list(a)
                
#------------------------------------------------------------------
