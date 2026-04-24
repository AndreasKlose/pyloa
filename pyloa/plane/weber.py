"""
    Module plane.weber of package pyloa:
    Methods for solving the Fermat-Weber problem.
"""
import numpy as np
from pyloa.util import euclid, all_circle_intersections

# Optimality tolerance
_EPSILON = 1.0E-04

# Tolerance to decide if current iterate falls on a customer point
_ZERO = 1.0E-4

#------------------------------------------------------------------

def _trivial_weber( Y, w ):
    """Return solution for at most two customer points"""
    # Only 1 customer point
    if Y.shape[0] < 2: return 0.0, Y[0], 0
    # Two customer points
    if w[0] < w[1]: return euclid(Y[0], Y[1])*w[1], Y[0], 0
    if w[0] > w[1]: return euclid(Y[0], Y[1])*w[0], Y[1], 0
    return euclid(Y[0],Y[1]), (Y[0]+Y[1])/2, 0   

#------------------------------------------------------------------

def _getGradient(X, Y, w):
    """
    Compute the gradient of current solution.

    Parameters
    ----------
    X : numpy array of two float specifying the current solution 
    Y : mx2 numpy array specifiying the m customer points
    w : numpy array of m customer weights (int or float)

    Returns
    -------
    (1) Object value of current solution X
    (2) Customer index on which current solution falls 
        (or -1 if X does not fall on any customer point)
    (3) Maximum distance of a customer point to X
    (4) Sum s(x) of modified weights w(i)/||X-Y(i)||
    (5) Gradient
    """
    # Obtain distances to current location and the objective value
    dist = np.fromiter((euclid(X,y) for y in Y),float)
    ObjVal = np.dot(w, dist)
    dmax = np.max(dist)

    # Obtain modified weights = weight/distance
    cIndex = np.argmin(dist)
    dmin = dist[cIndex]
    if dmin > _ZERO:
        cIndex = -1
        wid = w / dist
    else:
        dist[cIndex] = 1.0
        wid = w / dist
        wid[cIndex] = 0.0
        dist[cIndex] = 0.0

    # Compute gradient
    sx = wid.sum()
    gradient = X * sx - np.dot(wid, Y)
    return ObjVal, cIndex, dmin, dmax, sx, gradient

#------------------------------------------------------------------

def _weiszfeld(Y, w, l=1.0, screen='off', Xlst=None):
    """
    Computes optimal location X for a single Weber problem with 
    customer point locations Y and strictly positive customer 
    weights w.

    Parameters
    ----------
    Y : (numpy) array where each row i stores the 2-dimensional 
        vector Y(i) and Y(i) specifices customer i's coordinates.
    w : (numpy) array of customer weights.
    l : parameter "lambda" for adjusting the step lengths.
    screen: If 'on', the result of the current iteration is printed 
            to "stdout".
    Xlst : If not None, then Xlst is assumed to be an empty list 
           on input. On output it will contain the list of the 
           solutions X=(x1,x2) generated in each iteration.

    Returns
    -------
    (1) the optimal objective value, 
    (2) the optimal location as a numpy array of two float, 
    (3) the number of iterations
    """

    if Y.shape[0] <= 2: return _trivial_weber( Y, w )
      
    screenOn = (screen.lower() == 'on')
    if screenOn:
        print("---------------------------------------------------")
        print("Iter  Objective value  x-coordinate  y-coordinate")
        print("---------------------------------------------------")

    lam = min(max(1.0, l), 2.0)
    # Center of gravity is starting point
    X = np.dot(w, Y) / np.sum(w)
    itr = -1
    objVal = 0
    stop = (Y.shape[0] < 2)

    while not stop:
        itr += 1
        objVal, cIndex, _ , dmax, sx, gradient = _getGradient(X, Y, w)
        gnrm = np.linalg.norm(gradient)
        step = lam / sx
        if screenOn:
            print('{0:4d} {1:16.5f} {2:13.4f} {3:13.4f}'.\
                   format(itr, objVal, X[0], X[1]))
        if Xlst != None: Xlst.append( (X[0], X[1]) )
        if cIndex >= 0:
            # current iterate lies on customer point cIndex
            stop = (gnrm - w[cIndex] < _EPSILON)
            if not stop: step *= 1.0 - w[cIndex] / gnrm
        if not stop:
            # compute lower bound
            maxImprove = dmax * gnrm
            lowbnd = objVal - maxImprove
            stop = (lowbnd > _EPSILON) and (maxImprove/lowbnd < _EPSILON)
            # make the gradient step
            if not stop: X = X - step * gradient

    return objVal, X, itr

#------------------------------------------------------------------

def _drezner(Y, w, screen='off', Xlst=None):
    """
    Applies Drezner's method to compute the optimal facility's 
    location to the single Weber problem with customer point set Y 
    and customer weight vector w. All arguments (except l) are the 
    same as for function "Weiszfeld" above.
    
    Reference: Drezner, Z. (1992). A Note on the Weber Location 
    Problem. Annals of Operations Research 40:153–161.
    """
    if Y.shape[0] <= 2: return _trivial_weber( Y, w )
    
    screenOn = (screen.lower() == 'on')
    if screenOn:
        print("---------------------------------------------------")
        print("Iter  Objective value  x-coordinate  y-coordinate")
        print("---------------------------------------------------")

    X = np.dot(w, Y) / np.sum(w)
    itr = 0
    objVal = 0
    m = Y.shape[0]
    stop = (m < 2)

    while not stop:
        # Compute gradient et al. at current point
        objVal, k, dmin, dmax, sx, gradient = _getGradient(X, Y, w)
        gnrm = np.linalg.norm(gradient)
        if screenOn:
            print('{0:4d} {1:16.5f} {2:13.4f} {3:13.4f}'. \
                  format(itr, objVal, X[0], X[1]))
        if Xlst != None: Xlst.append( (X[0], X[1]) ) 

        # Check if eps-optimality is surely reached
        maximp = dmax * gnrm
        lowBnd = objVal - maximp
        stop = (lowBnd > _EPSILON) and (maximp/lowBnd < _EPSILON)
        if stop: break

        # If iterate falls on k-th customer point, check optimality
        # and if not optimal perform Ostresh step
        if k >= 0:
            if gnrm - w[k] < _EPSILON: break
            step = (1.0/sx) * (1.0 - w[k]/gnrm)
            X = X - step * gradient
        # If not on a customer point, do a step of Drezner's method
        else:
            weiszfp = X - gradient/sx
            # Get objective of Weiszfeld point & smallest customer distance
            dWeisz = np.fromiter((euclid(weiszfp,y) for y in Y),float)
            weiszObj = np.dot(w, dWeisz)
            weiszDmin = np.min(dWeisz)
            # Compute step length lambda as in Appendix of Chap.2 in the notes
            dstXWX = euclid(X, weiszfp)
            wdst = dstXWX * dstXWX * sx
            lam = 0.5 * wdst / (weiszObj - objVal + wdst)
            # Correct the step size lambda if too large
            if ( dmin < 0.01 ):
                lam=1.
            elif dmin > weiszDmin + _EPSILON:
                lam = min(lam, dmin / (dmin - weiszDmin))
            # Make a step of length lambda
            X = X - (lam/sx) * gradient
        itr += 1

    return objVal, X, itr
                       
#----------------------------------------------------------------

def solveWeber(Y,w,l=1.0,method='Drezner',screen='off', Xlst=None):
    """
    Solves instances of the single Weber problem with customer 
    point set Y and customer weight vector w. 

    Parameters
    ----------
    Y : mx2 numpy array of float
        Y[i] i stores the coordinates of the i-th customer point,
        i = 0,...,m-1
    w : numpy array of float or int
        customer weights
    l : float
        Parameter for adjusting step length in Ostresh's method,
        default value is 1, which is equivalent to Weiszfeld's method.
    method : str
        Solution method to apply, i.e. Weiszfeld, Ostresh or
        Drezner. Note that parameter l only need to be supplied 
        for Ostresh's method and then should be different from 1.  
    screen : str
        If 'on', the result of the current iteration is printed to "stdout"
    Xlst : None or an empty list
        If not None, then Xlst is assumed to be an empyt list on input. On 
        output it will contain the list of the solutions X=(x1,x2) generated 
        in each iteration.

    Returns
    -------
    obj : float
        The objective value of the optimal solution
    X : numpy array of two float
        The optimal location of the facility
    """        

    if method.lower() == 'weiszfeld' or method.lower() == 'ostresh':
        objVal, X, itr = _weiszfeld(Y, w, l, screen, Xlst)
    else:
        objVal, X, itr = _drezner(Y, w, screen, Xlst)
    if screen.lower() == 'on':
        print("--------------------------------------------------")
        print("Optimal objective value = ",objVal )
        print("Optimal location at ",X)
        print("Number of iterations    = ",itr)
        print("--------------------------------------------------")
    return objVal, X

#----------------------------------------------------------------

def limitedDist (Y, w, max_dist, screen='off' ):
    """
    Solves the single Weber problem with limited distances, which
    is to minimize the function g(X) where 
    
          g(X) = \sum_i w(i) min{ || X-Y(i)||, D(i) }
          
    Reference: Drezner, Z. (1984). The Planar Two-Center and 
    Two-Median Problems. Transportation Science 8, 351–361.
    
    Parameters
    ----------
    Y : mx2 numpy array of float
        Y[i] i stores the coordinates of the i-th customer point,
        i = 0,...,m-1
    w : numpy array of float or int
        customer weights
    max_dists : list or numpy array of float or int 
        max_dists[i] is the limited distance that applies for
        customer point i.
    screen : str
        If 'on', the result of the current iteration is printed to "stdout"
        
    Returns
    -------
    objval : float
         Objective value of the optimal solution
    X : numpy array of two float
        The coordinates of the optimal location
    """
    m = len(Y)
    itr = 0
    talk = screen.lower() == 'on'
    if talk:
        print('-'*40)
        print('Iterat  Objective value  Best objective')
        print('-'*40)
   
    # Objective function g(X) 
    g = lambda X : sum( w[i]*min(euclid(X, Y[i]),max_dist[i]) for i in range(m) )
    
    # best objective value and location    
    best_objv = np.infty 
    X_best = np.zeros(2,dtype=float)
        
    # Dictionary of sets S for which Fermat-Weber problem already solved
    solved = dict() 
    
    # Simple function checking if improved solution found
    def check_improve( X ):
        nonlocal best_objv, itr
        itr += 1 
        objv = g(X)
        if objv < best_objv:
            best_objv = objv 
            X_best[:] = X[:]
        if talk: print('{0:6d}  {1:15.2f}  {2:14.2f}'.format(itr, objv, best_objv))
        
    # Function below checks if S in "solved" and if not adds it to "solved"
    def in_solved( new_S ):
        num = len(new_S) 
        SS  = set(new_S)
        try: 
            for S in solved[num]: 
                if S == SS: return True  
            solved[num].append(SS)
        except:
            solved[num] = [SS] 
        return False 
       
    # Let C(i) be circle of radius max_dist[i] centered at Y[i].
    # Function below returns true if C(j) included in C(i)
    circle_in_circle = lambda i,j : max_dist[j] < max_dist[i] and \
                    euclid(Y[i], Y[j]) + max_dist[j] < max_dist[i]
    
    # Enumerate all sets S(i) where S(i) = {j : C(j) \subseteq C(i)}.
    # Thereby C(i) is the circle of radius max_dist[i] with center Y[i].
    for i in range(m):
        S = list( filter( lambda j : circle_in_circle(i,j), range(m) ) )
        if len(S)==0:
            check_improve( Y[i] )
        else:
            S.append(i)
            if not in_solved( S ):
                _ , X  = solveWeber( Y[S], w[S] )
                check_improve( X )
     
    # For each intersection point P(i,j) of circles C(i) and C(j) check the
    # 4 sets S(i,j), S(i,j)\{i}, S(i,j)\{j} and S(i,j)\{i,j}, whereby
    # S(i,j) comprises all indices k where C(k) contains P(i,j)
    in_circle = lambda P,k : euclid(P,Y[k]) < max_dist[k]+_ZERO 
    all_is_points = all_circle_intersections( Y, max_dist )
    for num in all_is_points: 
        i, j = num // m, num % m  
        for P in all_is_points[num]:
            # Check set S(i,j) = {k : P(i,j) in C(k)}
            S = list( filter( lambda k : in_circle(P,k), range(m) ) )
            if not in_solved( S ):
                _, X = solveWeber( Y[S], w[S] ) 
                check_improve( X )
            if len(S) <= 2: continue
            # Check set S(i,j)\{i} 
            ii = S.index(i)
            S[ii], S[-1] = S[-1], i
            S_i = S[:-1]
            if not in_solved( S_i ):  
                _, X = solveWeber( Y[S_i], w[S_i] )
                check_improve( X ) 
            # Check set S(i,j)\{j}
            jj = S.index(j)
            S[jj], S[-2], S[-1] = S[-2], i, j
            S_j = S[:-1]
            if not in_solved( S_j ):
                _, X = solveWeber( Y[S_j], w[S_j] )
                check_improve( X )
            # Check set S(i,j)\{i,j}
            if len(S) > 3:
                S_ij = S[:-2]
                if not in_solved( S_ij ):
                    _, X = solveWeber( Y[S_ij], w[S_ij] )
                    check_improve( X )
                
    return best_objv, X_best 

#----------------------------------------------------------------
