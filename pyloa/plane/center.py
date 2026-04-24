"""
    Module plane.center of package pyloa:
    Methods for the 1-center problem in Euclidian plane.
"""

import numpy as np
from pyloa.util import euclid

_EPSI = 1.0E-6
_ZERO = 1.0E-6

#------------------------------------------------------------------

def _sqeuc( A, B ):
    """
    Returns squared Euclidian distance between points A and B
    """
    diff = A-B
    dst2 = np.dot(diff,diff)
    return dst2

#------------------------------------------------------------------

def _acute( P, Q, R ):
    """
    Checks if triangle spanned by points P, Q, R is not obtuse.
    Triangle is not obtuse if c^2 <= a^2 + b^2, where c is
    length of longest side.
    """
    PQ = _sqeuc(P, Q)
    PR = _sqeuc(P, R)
    QR = _sqeuc(Q, R)
    ML = max( PQ, PR, QR )
    return not ( 2*ML > PQ + PR + QR )

#------------------------------------------------------------------

def _2points( Y, w ):
    """
    Returns radius and center for the weighted or unweighted
    1-center problem in case of just two points Y1 and Y2 of
    weights w1 and w2.
    """
    X = (w[0]*Y[0]+ w[1]*Y[1])/(w[0]+w[1])
    r = w[0]*euclid( X, Y[0] )
    return r, X

#------------------------------------------------------------------

def _3circle( P, Q, R ):
    """
    Returns radius and center of the circle enclosing the triangle 
    spanned by the three points P, Q and R. The triangle is assumed to 
    be actute or right-angled.
    
    Parameters
    ----------
    P, Q, R : 3 iterables of floats of length 2
        The three points on the circle's boundary
            
    Returns
    -------
    r: float
        The radius of the circle
    C: numpy array of 2 float
        The coordinates of the circle's center
    """
    (x2,y2) = Q-P
    (x3,y3) = R-P
    LQ = x2*x2 + y2*y2
    LR = x3*x3 + y3*y3
    D = x2*y3 - x3*y2
    C = np.array( ( y3*LQ - y2*LR, x2*LR - x3*LQ ) )/(2*D)
    r = np.linalg.norm(C)
    C[0] += P[0]
    C[1] += P[1]
    return r, C

#------------------------------------------------------------------

def _get_weight_circle( Y1, Y2, w1, w2 ):
    """
    Returns radius and center of the circle whose boundary is the
    set of points showing equal weighted distance to the two
    points Y1 and Y2, where 0 < w1 < w2
    """
    r = w1/w2
    radius = euclid(Y1,Y2)*r/(1-r**2)
    C = (Y2 - Y1*r**2)/(1-r**2)
    return radius, C

#------------------------------------------------------------------

def _line_circle_intersect(P,Q,R,w_PQ,w_R):
    """
    Intersects the perpendicular bisector of the side connecting
    the points P and Q of equal weight w_PQ with the circle 
    describing the set of points showing equal weighted distance to 
    point Q and point R (w_PQ must be different from w_R).
    """
    if w_PQ < w_R:
        radius, C = _get_weight_circle(Q, R, w_PQ, w_R)
    else: 
        radius, C = _get_weight_circle(R, Q, w_R, w_PQ)
        
    if abs(P[1]-Q[1]) < _ZERO:
        # Case 1: bisector parallel to y-axis
        x1 = (P[0]+Q[0])/2
        x2 = x1
        delta = np.sqrt( radius**2 - (x1-C[0])**2 )
        y1 = C[1] + delta
        y2 = C[1] - delta
    elif abs(P[0]-Q[0]) < _ZERO:
        # Case 2: bisector parallel to x-axis
        y1 = (P[1]+Q[1])/2
        y2 = y1
        delta = np.sqrt( radius**2 - (y1-C[1])**2 )
        x1 = C[0] + delta
        x2 = C[0] - delta
    else:
        # Let y = a + bx describe the perpendicular bisector on P-Q
        b = (P[0]-Q[0])/(Q[1]-P[1])
        a = ( (P[1]+Q[1]) - b*(P[0]+Q[0]) )/2 
        # Solve system (x-C.x)^2 + (y-C.y)^2=radius^2     and y=a+bx.
        # Inserting y=a+bx in first equation gives quadratic equation
        # alpha*x^2 + beta*x + gamma, where:
        alpha = 1 + b**2
        beta = 2*(b*(a - C[1]) - C[0])
        gamma = C[0]**2 + (a-C[1])**2 - radius**2
        delta = np.sqrt( beta**2 - 4*alpha*gamma)
        x1 = (delta - beta)/(2*alpha)
        x2 = -(delta+beta)/(2*alpha)
        y1 = a + b*x1
        y2 = a + b*x2
    
    # Two candidate solutions remain and both show equal weighted 
    # distance to points P and Q as well as Q and R, hence to all 
    # three points. Select then the one inside the triangle, which 
    # must be the one yielding smallest weighted distance.    
    C1 = np.array((x1,y1))
    C2 = np.array((x2,y2))
    r1 = w_PQ*euclid(P, C1)
    r2 = w_PQ*euclid(P, C2)
    if r1 < r2: return r1, C1
    return r2, C2

#------------------------------------------------------------------

def _circle_circle_intersect(P,Q,R,w_P,w_Q,w_R):
    """
    Determines weighted 1-center solution for three points P,Q,R
    with weights w_P < w_Q < w_R by intersecting the two circles
    C(P,Q) and C(Q,R) whose boundary is the set of points (x,y)
    respectively showing equal weighted distance to P and Q as
    well as Q and R.
    """
    # Get the two circles of points with equal weighted distance
    # to (P,Q) and (Q,R), respectively
    r1, C1 = _get_weight_circle(P,Q,w_P,w_Q)
    r2, C2 = _get_weight_circle(Q,R,w_Q,w_R)
   
    # Find the intersections I1 and I2 of these two circles
    # (cf. Appendix to Chap. 4, Lecture Notes on Location Planning)
    d = euclid(C1,C2)
    K = np.sqrt( ((r1+r2)**2 - d**2)*(d**2 - (r1-r2)**2) )/4
    ax = (C1[0]+C2[0])/2 + ( (C2[0]-C1[0])/(2*d**2) )*(r1**2-r2**2)
    bx = 2*(C2[1]-C1[1])*K/d**2
    ay = (C1[1]+C2[1])/2 + ( (C2[1]-C1[1])/(2*d**2) )*(r1**2-r2**2)
    by = 2*(C2[0]-C1[0])*K/d**2
    I1 = np.array( (ax + bx, ay - by) )
    I2 = np.array( (ax - bx, ay + by) )
    
    # The center solution is the point inside the triangle
    # spanned by P, Q, R and should be the point showing
    # smaller weighted distance to these three points.
    rad1 = w_P*euclid(P,I1)
    rad2 = w_P*euclid(P,I2)
    if rad1 < rad2: return rad1, I1
    return rad2, I2

#------------------------------------------------------------------

def _w_3circle( Y, w ):
    """
    Solves the weighted 1-center problem for three points given
    by Y[0],Y[1] and Y[2] with weights w[0], w[1], w[2]. It is
    thereby assumed that weighted distance from the center to
    all these three points can be made equal.
    """
    if w[0]==w[1] and w[1]==w[2]: return _3circle(Y[0],Y[1],Y[2])
    
    order = np.argsort(w[:3])
    w_P, w_Q, w_R = w[order[0]], w[order[1]], w[order[2]]
    P, Q, R = Y[order[0]], Y[order[1]], Y[order[2]]
    
    if w_P==w_Q: return _line_circle_intersect(Q,P,R,w_Q,w_R)
    if w_Q==w_R: return _line_circle_intersect(Q,R,P,w_Q,w_P)
    return _circle_circle_intersect(P,Q,R,w_P,w_Q,w_R)
    
#------------------------------------------------------------------

def _PD_alg( Y, w, screen='off', Xlst=None, normalize=False ):
    """
    Primal-dual iterative method for solving the 1-center problem
    in Euclidian plane with Euclidian distances. The method
    shows severe numerical problems as the Lagrangian multipliers 
    corresponding to points not relevant for computing the center 
    quickly approach very small values. The method should thus 
    only be used if the number of points is very small 
    (4 at most 5)

    Parameters
    ----------
    Y : numpy mx2 array of float
        m customer points in Euclidian plane
    w : numpy array of m int or float 
        positive customer weights
    screen : str, optional 
        If 'on', then progress of the algorithm is displayed on screen.
    Xlst : None or a list of pairs of float, optional
        If not None, the intermediate center solutions are appended
        to the list Xlst.
    normalize : bool 
        If true, the weights are normalized by dividing by there
        mean. This is recommended if the weights are not yet
        normalized and are relatively large (in particular when
        squared)

    Returns
    -------
    UB : float
        "radius" of the computed center location
    X  : numpy array of two float
        coordinates of the center found
    """
    m = Y.shape[0]
    X = np.zeros(2)
    wmean = w.sum()/m if normalize else 1.0
    ws = np.square( w/wmean )
    
    # Initial Lagrangian multiplier values
    u = np.ones(m)/m
    itr = 0
    screenOn = screen.lower() == 'on'
    if screenOn:
        print("------ Computing 1-center in plane -----")
        print("Iter  Lower Bound  Upper Bound")

    # Iterative part
    while True:
        itr += 1
        # Modifiy squared weights with Lagrangian multipliers
        wu = ws*u
        # Get primal solution as gravity center 
        X = np.dot(wu,Y) / np.sum(wu)
        if Xlst != None: Xlst.append( (X[0], X[1]) ) 
        # Squared distance of customer points to X and primal objective
        dist=np.fromiter((_sqeuc(X,Y[i]) for i in range(m)),float)
        UB = np.max( dist*ws )
        # Value of Lagrangian dual function at u 
        LB = np.dot(wu, dist)
        # Check if gap between UB and LB is small enough
        if ( UB - LB )/LB < _EPSI: break
        # Adjust the Lagrangian multipliers
        u = np.maximum(wu*dist/LB,_EPSI)
        if screenOn:
            print('{0:4d}  {1:11.4f}  {2:11.4f}'.format(itr,LB,UB))
    
    if screenOn:
        print("------------------------------------------")

    return np.sqrt( UB )*wmean, X                 

#------------------------------------------------------------------

def _getRadius( Y, w, X, b=None ):
    """
    Computes radius of a center point in X.
    
    Parameters
    ----------
    Y : numpy mx2 array of float
        m customer points in Euclidian plane
    w : numpy array of m int or float
        positive customer weights
    X : numpy array of two float 
        coordinates of center point
    b : list of points, optional
        if b is present, it need be a list of points. 
        The largest weighted distance to points in b 
        serves as lower bound.

    Returns
    -------
    wdist : float
        radius of the solution
    outer : int
        index i of a point Y[i] showing largest weighted 
        distance to X 
    bdist : float (only returned if b is not None)
        if b is not None, the largest weighted distance 
        to points Y in b.
    """
    m = Y.shape[0]
    wdist=np.fromiter( (euclid(X,Y[i])*w[i] for i in range(m)), float ) 
    outer=np.argmax( wdist )
    if b == None:
        return wdist[outer], outer
    else:
        return wdist[outer], outer, np.max(wdist[b])

#------------------------------------------------------------------

def growRadius ( Y, w, screen='off', Xlst = None ):
    """
    Solves the weighted 1-center problem in plane by growing the
    "radius" until all points are covered. The method starts
    by selecting two initial points for which the 1-center problem
    is solved. Then additional uncovered points are included and
    the adjusted 1-center problem on a small subset of points is
    solved. The method continues this way until all points are
    covered.

    Parameters
    -----------
    Y : numpy mx2 array of float
        m customer points in Euclidian plane
    w : numpy array of m int or float
        positive customer weights
    screen: str
        if 'on', information on solution process is printed
    Xlst : list of tuple of float or None
       if not none, the points found in each "iteration"
       are stored in this list.

    Returns
    -------
    UB : float
        "radius" of the computed center location
    X  : numpy array of two float 
        coordinates of the center as a 
    """
    screenOn = screen.lower() == 'on'
    if screenOn:
        print("------ Computing 1-center in plane -----")
        print("Iter  Lower Bound  Upper Bound  Current Center")
        
    # Normalize the weights by dividing with there mean
    wsum = w.sum()
    wmean = wsum/len(Y)
    w = w/wmean
    
    # Compute center of gravity
    X = np.dot(w,Y) / np.sum(w)
   
    # 1st point is the one showing largest weighted distance to X
    _, out = _getRadius( Y, w, X )
    basis = [out]

    # 2nd point shows largest weighted distance to the 1st
    _, out = _getRadius( Y, w, Y[basis[0]] )
    basis.append( out )

    # Initial center is gravity center of the selected two points
    X = np.dot( w[basis],Y[basis] )/ np.sum( w[basis] )
    UB, out, LB = _getRadius( Y, w, X, basis )

    if Xlst != None: Xlst.append( (X[0],X[1]) )
    
    # Continue as long as UB exceeds LB
    itr = 0
    while ( UB-LB )/max(LB,1.0) > _EPSI:        
       
        itr += 1
        # Print current solution
        if screenOn:
            print('{0:4d}  {1:11.4f}  {2:11.4f}  {3:.5f}/{4:.5f}'.\
                  format(itr,LB,UB,X[0],X[1]))

        # Remove points strictly covered from the "basis"
        for b in basis:
            if euclid(Y[b],X)*w[b] < LB*(1.0 - _EPSI): basis.remove(b)  
        
        # Include the outer point in the basis
        basis.append( out )
        
        # Find 1-center for the points in the basis
        LB, X = _PD_alg ( Y[basis], w[basis] )

        # Update upper and lower bound
        UB, out, LB = _getRadius( Y, w, X, basis )

        if Xlst != None: Xlst.append( (X[0],X[1]) )

    if screenOn:
        print('{0:4d}  {1:11.4f}  {2:11.4f}  {3:.5f}/{4:.5f}'.\
              format(itr,LB,UB,X[0],X[1]))
        print("------------------------------------------")

    return UB*wmean, X

#------------------------------------------------------------------

def elzinga_hearn( Y, screen='off', Xlst=None ):
    """
    Elzinga and Hearn's method to determine the minimal covering
    circle of m points in Euclidian plane.
    
    Reference: Elzinga, J. and Hearn, D. W. (1972).  Geometrical 
    Solutions of Some Minimax Location Problems.Transportation 
    Science 6:370–394.
  
    Parameters
    ----------
    Y : numpy mx2 array of float
        m customer points in Euclidian plane
    screen : str, optional 
        If 'on', then progress of the algorithm is displayed on screen.
    Xlst : None or a list of pairs of float, optional
        If not None, the intermediate center solutions are appended
        to the list Xlst.
  
    Returns
    -------
    R : float
        "radius" of the computed center location
    X  : numpy array of two float
        coordinates of the center found
    """
    m = Y.shape[0]
    if m <= 1: return 0.0, Y[0]
    
    # Function returning on which side of a plane a point p is
    absv, slope = 0.0, 0.0
    H = lambda p : p[1] - ( absv + slope*p[0] )
    
    screenOn = screen.lower() == 'on'
    if screenOn:
        print("----- Computing minimal covering circle ------")
        print("Iter  Lower Bound  Upper Bound  Current Center")
    
    # Center of gravity 
    X = Y.sum(axis=0) / m
    if m == 2: return euclid(X,Y[0]), X
    
    w = np.ones(m,dtype=int)
    basis = [0,0,0]
    
    # Get most distant point to X and most distant one to the former
    _, basis[0] = _getRadius( Y, w, X )
    _, basis[1] = _getRadius( Y, w, Y[basis[0]] )
    bdim = 2
    
    # Main loop: Repeat as long as Circle(Basis) does not cover all points
    itr = 0
    while True:
        # Determine radius and circle for points Y[basis]
        if bdim==2:
            RB, X  = _2points(Y[basis[:2]], [1,1]) 
        else: 
            RB, X = _3circle(Y[basis[0]],Y[basis[1]],Y[basis[2]])
        R, out = _getRadius( Y, w, X )
        if screenOn:
            print('{0:4d}  {1:11.4f}  {2:11.4f}  {3:.5f}/{4:.5f}'.format(itr,RB,R,X[0],X[1]))
        if R <= RB*(1+_EPSI): break
        if bdim==3:
            # Current circle determined by three points
            D = Y[out]
            # Sort the three basis points by decreasing distance to D
            basis.sort(key = lambda i : euclid(Y[i],D),reverse=True)
            A = Y[basis[0]]
            # Get hyperplane through points A and X 
            slope = (A[1]-X[1])/(A[0]-X[0])
            absv = A[1] - slope*A[0]
            # Second basis point is opposite to D regarding hyperplane
            if np.sign( H(D) ) <= 0.0:
                if np.sign(H(Y[basis[2]])) > 0.0: basis[1]=basis[2]
            else:
                if np.sign(H(Y[basis[2]])) < 0.0: basis[1]=basis[2]
                 
        # If non-acute triangle, basis consists of point outside current circle
        # and the one most distant to it
        basis[2] = out
        bdim = 3 
        A, B, C = Y[basis[0]], Y[basis[1]], Y[basis[2]] 
        if not _acute(A, B, C):
            bdim = 2
            if euclid(B,C) > euclid(A,C): 
                basis[0], basis[1] = basis[1], basis[2]
            else:
                basis[1] = basis[2]    
        itr += 1
            
    return R, X 

#------------------------------------------------------------------

def charalambous(Y, w, screen='off', Xlst=None):
    """
    Charalambous' (1982) method for solving the weighted 1-centre
    problem in the plane.
    
    Reference: Charalambous, C. (1982). Extension of the Elzinga-Hearn
    Algorithm to the Weighted Case. Operations Research 30:591–594.
    
    Parameters
    ----------
    Y : numpy mx2 array of float
        m customer points in Euclidian plane
    w : numpy array of m int or float
        positive customer weights
    screen: str
        if 'on', information on solution process is printed
    Xlst : None or a list of pairs of float, optional
        If not None, the intermediate center solutions are appended
        to the list Xlst.
  
    Returns
    -------
    UB : float
        "radius" of the computed center location
    X  : numpy array of two float
        coordinates of the center as a 
    """
    
    def initialCenter( w ):
        """ Determine initial solution """  
        # center of gravity:
        X = np.dot(w,Y) / np.sum(w)  
        # point showing largest weighted distance to X
        _, out = _getRadius( Y, w, X ) 
        basis = [out]
        # point showing largest weighted distance to the 1st
        _, out = _getRadius( Y, w, Y[basis[0]] )
        basis.append( out )
        # Initial center is 2-point solution with basis "basis"
        X = np.dot( w[basis],Y[basis] )/ np.sum( w[basis] )
        return X, basis
        
    def new_basis_3points( w, basis, out ):
        """ Find solution with a basis that containing point 'out' and
            one or two points of the current 2-point basis """
        covered = False
        # Check all 2-point solutions covering points Y[out]
        # and a single point from the current basis
        B = basis.copy()
        R = np.infty 
        for i in range(2):
            j, B[i] = B[i], out 
            r, x = _2points( Y[B], w[B] )
            if w[j]*euclid(x, Y[j]) <= r*(1 + _EPSI) and r < R:
                covered = True 
                R, X = r, x
                basis[0], basis[1] = B[0], B[1]
            B[i] = j
        if covered: return R, X, 2
        
        # If no two point basis it need to be the 3-point one
        basis[2] = out
        R, X = _w_3circle( Y[basis], w[basis] )
        return R, X, 3        
    
    def new_basis_4points( w, basis, out ):
        """ Find solution for the 1-centre problem with 4 customer points
            given by the set basis+{out}. The basis for the solution need 
            thereby to contain the point 'out' together with one or two 
            points of the current 3-point basis """ 
        R = np.infty
        B = basis.copy()
        b = [0,0,0]
        bdim = 0
        for i in range(3):
            b[:] = basis[:]
            # Exclude point j=B[i] and include out
            j = b[i]
            b[i], b[-1] = b[-1], b[i] 
            # Find solution to 3-points problem {B[0], B[1], out}
            r, x, bsize = new_basis_3points( w, b, out )
            # Check if this also covers point j
            if w[j]*euclid(x, Y[j]) <= r*(1+_EPSI) and r < R:
                R, X, bdim = r, x, bsize
                B[:] = b[:] 
        # Return the solution and the basis' dimension
        basis[:] = B[:]
        return R, X, bdim
             
    #--------------------
    # Begin main routine 
    #--------------------    
    itr = 0
    screenOn = screen.lower() == 'on'
    if screenOn:
        print("------ Computing 1-center in plane -----")
        print("Iter  Lower Bound  Upper Bound  Current Center")
    
    # Get weights as flat numpy array
    if w is None: w = np.ones(len(Y),dtype=int)
    
    # Create initial solution 
    X, basis = initialCenter( w )
    if Xlst != None: Xlst.append( (X[0],X[1]) )
    UB, out, LB = _getRadius( Y, w, X, basis )
    if screenOn:
        print('{0:4d}  {1:11.4f}  {2:11.4f}  {3:.5f}/{4:.5f}'.format(itr,LB,UB,X[0],X[1]))
    
    # Main loop
    bdim = len(basis)
    if bdim == 2: basis.append(-1)
    while ( UB-LB )/max(LB,1.0) > _EPSI: 
        itr += 1
        if bdim==2:
            LB, X, bdim = new_basis_3points( w, basis, out )
        else:
            LB, X, bdim = new_basis_4points( w, basis, out )
        if not Xlst is None: Xlst.append( (X[0],X[1]) )
        UB, out = _getRadius(Y, w, X )
        if screenOn:
            print('{0:4d}  {1:11.4f}  {2:11.4f}  {3:.5f}/{4:.5f}'.format(itr,LB,UB,X[0],X[1]))
    
    if Xlst is None: return UB, X 
    return UB, X, Xlst
    
#------------------------------------------------------------------

def welzl( Y, w, screen='off' ):
    """
    Welzl's recursive algorithm for computing the covering circle.
    The algorithm's simple idea is as follows. Let P be the list
    of points we have to cover. Let B denote the basis, that is
    the set of (up to three) points whose covering circle determines
    the complete solution and which thus lie on the circle's boundary. 
    Initially, let B be empty. Choose then a point p from P and let S 
    be the solution for point set P-{p}. If point p is covered we're 
    done. Otherwise, point p need to be one of those points on the 
    circle's boundary and we add p to B. If B comprises three points, 
    the circle is uniquely determined. Otherwise, we try to find the 
    smallest enclosing circle of point set P-{p} with points B+{p} on 
    the boundary. 
    
    Reference: Emo Welzl (1991). Smallest enclosing disks (balls and 
    ellipsoids). In Maurer, H., ed., New Results and New Trends in 
    Computer Science. Lecture Notes in Computer Science, Vol. 555, 
    Springer-Verlag, pp. 359–370, doi:10.1007/BFb0038202.

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
    UB : float
        "radius" of the computed center location
    X  : numpy array of two float
        coordinates of the center as a 
    """
    itr = 0
    talk = screen.lower()=='on'
    
    if w is None: w = np.ones(len(Y),dtype=int)
              
    def _trivial_circle( B ):
        """
        Return weighted center solution of up to three points
        Y[B] such that all points Y[B] show equal weighted distance
        to the center point.
        """
        n = len(B)
        if n==0: return 0.0, np.array( (0.0,0.0) )
        if n==1: return 0.0, Y[B[0]]
        if n==2: return _2points( Y[B], w[B] )
        if n==3: return _w_3circle(Y[B], w[B])
        
    def _welzl_recursion( P, B, n):
        """
        Find recursively the smallest enclosing circle of the
        n first points in P such that points in B are on the
        circle's boundary.
        """
        nonlocal itr
        itr += 1
        
        if (n==0) or (len(B)==3):
            return _trivial_circle( B )
        
        # Find solution for first n-1 points with boundary B
        p = P[n-1]
        r, X = _welzl_recursion( P, B.copy(), n-1 )
        
        # Check if point Y[p] is covered
        covered = w[p]*euclid(X, Y[p] ) <= (1.0 + _EPSI)*r 
        
        # If point Y[p] is covered, we found the solution for
        # the minimal circle of the first n points with points
        # Y[B] on the boundary.
        if covered: 
            if talk: 
                print('{0:4d}  {1:14.4f}  {2:.5f}/{3:.5f}'.format(itr,r,X[0],X[1]))
            return r, X
        
        # If Y[p] is not covered, it must be on the boundary.
        # Hence, find minimal circle covering the first n-1
        # points with points Y[B] and Y[p] on the boundary.
        B.append(p)
        return _welzl_recursion(P, B.copy(),n-1)
    
    # Start of procedure welzl    
    n = len(Y)
    P = list(np.random.permutation(range(n)))
    
    if talk:
        print("------ Computing 1-center in plane -----")
        print("Iter  Current Radius  Current Center")
    
    return _welzl_recursion(P, [], n)
         
