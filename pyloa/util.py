"""
    Util - Some helper functions/routines used in package pyloa.
"""
import numpy as np
import geopy
import math

from geopy.distance import EARTH_RADIUS as __R # gives 6371.009km
if geopy.__version__ < '1.14': 
    from geopy.distance import vincenty
else:
    from geopy.distance import geodesic
    
from itertools import combinations

_ZERO = 1.0E-06
    
#-----------------------------------------------------------------------

def euclid( X, Y ):
    """
    Returns Eucldean distance between two points X and Y.

    Parameters
    ----------
    X, Y : two numpy arrays of float of dimension 2
    
    Returns
    -------
    euclid : float
        Euclidean distance between the two points.
    """
    return np.linalg.norm( X-Y)

#-----------------------------------------------------------------------

def geo_dist( p1, p2 ):
    """
    Uses geopy to obtain the geodesic (that is Vincenty) distance 
    in km between two points specified by longitude and latitude.

    Parameters
    ----------
    p1, p2 : two numpy arrays of float of dimension 2
    
    Returns
    ------- 
    geo_dist : float
        The geodesic distance between the two points.
    """
    if geopy.__version__ < '1.14': return( vincenty(p1,p2) )
    return( geodesic(p1,p2).km )

#----------------------------------------------------------------------

def lla2xy ( lo, la, origin ):
    """
    Converts a point given by longitude and latitude to a cartesian 
    point relative to the longitude and latitude point of a given 
    origin.

    Parameters
    ----------
    lo, la: two floats
        longitude and latitude of point to convert
    origin: numpy array of two float
        longitude and latitude of the origin point
    
    Returns
    -------
    lla2xy: list of two float
        The cartesian/euclidean coordinates of the point.
    """
    # convert north-south distance from degrees to radians
    y = __R*(la-origin[1])*math.pi/180.0;

    # for x-coordinate, use distance along a line of latitude from the
    # point's longitude to the "origin'" longitude
    x = __R*(lo - origin[0])*math.pi/180.0*math.cos(origin[1]*math.pi/180.0)

    return( [x,y] )

#----------------------------------------------------------------------

def xy2lla( x, y, origin ):
    """
    Inverse function of lla2xy. Given the Euclidian point (x,y),
    return longitude and latitude by reversing above transformation.
    
    Parameters
    ----------
    x : float  
        x-coordinate of the point
    y : float  
        y-coordinate of the point
    origin : numy array or list of two floats  
        longitude and latitude of the origin point
           
    Returns
    -------
    lo, la : two floats
        longitude and latitude corresponding to (x,y)
    """
    la = origin[1] + y*180/__R/math.pi
    lo = origin[0] + x*180/__R/math.pi/math.cos(math.pi*origin[1]/180)
    return lo, la
    
#----------------------------------------------------------------------

def all_two_parts( Y ):
    """
    Creates all possible divisions of the customer point set {y(i)} into
    two subsets such that the convex hulls of the two subsets do not
    overlap.
    
    Parameters
    ----------
    Y : mx2 numpy array of float
        Y[i] specifies the customer i's coordinates, i=0,...,m-1
    
    Returns
    -------
    all_parts : dict of pairs of lists of int
        all_parts[num] is a pair of two lists of int such that the
        number of elements in the smaller of the two lists equals the
        number num. Note that num is at most (m-1)/2 if m is odd
        and m/2 if m is even. If all_parts[num] = P, then the 
        division of the customer set in the two subsets is P[0] and
        P[1]. The smaller of the two sets is thereby always listed
        first, i.e. len(P[0]) < len(P[1]). In both parts are of
        same length, the part containing the smallest customer index
        is listed first, i.e. min(P[0]) < min(P[1]) if len(P[0])==
        len(P[1]).
    """
    all_partits = dict()

    def create_partition( y1,y2, noise ):
        """
        Create the two possible partitions using separating line through
        points y1--y2 (rotated slightly around the lines middle point)
        """
        # Determine line a*y[1] + b*y[0] = c through points y1 and y2
        a = y2[0] - y1[0]
        b = y1[1] - y2[1]
        c = y2[0]*y1[1] - y1[0]*y2[1]
        
        alpha = ( c+noise*(y1[1] + y2[1])/2, a+noise, b ) if abs(b) > 1.0E-6 else \
                ( c+noise*(y1[0] + y2[0])/2, a, noise )
              
        left = lambda y : alpha[1]*y[1] + alpha[2]*y[0] > alpha[0]
        I1 = list( filter( lambda i : left(Y[i]), range(len(Y)) ) )
        I2 = list( filter( lambda i : not left(Y[i]), range(len(Y))) )
        m1, m2 = len(I1), len(I2)
        if m1 < m2: return I1, I2
        if m2 < m1: return I2, I1 
        if min(I1) < min(I2): return I1, I2 
        return I2, I1 
    
    def add_partition( I1, I2 ):
        # Checks if the customer set partition (I1,I2=I\I1) has already been generated
        num = len(I1)
        if num > 0:
            try: 
                found = False 
                for P in all_partits[num]: 
                    found = I1 == P[0]
                    if found: break
                if not found: all_partits[num].append((I1,I2))
            except:
                all_partits[num] = [(I1,I2)]  
        
        
    for y1, y2 in combinations(Y,2):
        for eps in (_ZERO,-_ZERO ):
            I1, I2 = create_partition( y1, y2, eps )
            add_partition(I1, I2)
     
    return all_partits 

#----------------------------------------------------------------------

def all_circle_intersections( Y, max_dists ):
    """
    Let C(i) be the circle of radius max_dists[i] around customer point
    Y[i]. This function computes the up to two intersection points 
    between circles C(i) and C(j) for every customer point pair.
    
    Parameters
    ----------
    Y : mx2 numpy array of float
        Y[i] gives the two coordinates of customer point i
    max_dists : list or numpy array of float or int 
        max_dists[i] is the radius of the circle C(i) around point 
        Y[i]. Usually this radius refers to a maximal distance for 
        customer i.
        
    Returns
    -------
    all_intersects : dict
         Dictionary where the key refers to the customer pair and
         the value to the up to two intersection points. If circles
         C(i) and C(j) intersect at the two points P and Q, then
         all_intersects[num] is the tuple (P,Q) consisting of
         the two points P and Q, which are in turn tuples of
         two coordinates. Thereby i = num//m and j=num % m. 
         If C(i) and C(j) have only a single intersection point P, 
         then all_intersects[num] is a list (and not a tuple) 
         consisting of the single point P. 
    """
    
    def _can_intersect(i,j):
        # Returns true if circles C(i) and C(j) have intersection
        d_ij = euclid(Y[i],Y[j])
        # Check if intersection of circle areas is empty
        if d_ij > max_dists[i] + max_dists[j]: return False
        # Check if one circle is contained in the other
        r1, r2 = max_dists[i], max_dists[j]
        if r1 > r2 : r1, r2 = r2, r1
        if not d_ij + r1  > r2 : return False 
        return True
    
    def _circle_intersect(i,j):
        # Return the two or the single intersection point(s) of 
        # circles C(i) and C(j)
        r1, C1 = max_dists[i], Y[i]
        r2, C2 = max_dists[j], Y[j]
        # Find intersections (cf. Appendix to Chap. 4)
        d = euclid(C1,C2)
        ax = (C1[0]+C2[0])/2 + ( (C2[0]-C1[0])/(2*d**2) )*(r1**2-r2**2)
        ay = (C1[1]+C2[1])/2 + ( (C2[1]-C1[1])/(2*d**2) )*(r1**2-r2**2)
        if r1 + r2 - d < _ZERO: return [(ax,ay)]
        K  = np.sqrt( ((r1+r2)**2 - d**2)*(d**2 - (r1-r2)**2) )/4
        bx = 2*(C2[1]-C1[1])*K/d**2
        by = 2*(C2[0]-C1[0])*K/d**2
        I1 = (ax + bx, ay - by) 
        I2 = (ax - bx, ay + by)
        return (I1,I2)
        
    m = len(Y)
    all_intersects = dict()
    for i, j in combinations(range(m),2):
        if _can_intersect(i,j):
            all_intersects[i*m+j] = _circle_intersect(i,j)
            
    return all_intersects
#----------------------------------------------------------------------
