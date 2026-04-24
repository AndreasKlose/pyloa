"""
    Module plane.parser of package pyloa:
    Read data of a planar location problem.
"""
import numpy as np
import pandas as pd
import os
import tsplib95 as tsp

from geopy.geocoders import Nominatim
from pyloa.util import lla2xy


_lo = None     # list of longitude data
_la = None     # list of latitude data
_names = None  # list of customer/city names
 
#----------------------------------------------------------------------

def read_points( data_file ):
    """
    Read data for a planar location problem either from a csv file
    or from a TSP data file from TSPlib (library of TSP test instances).
    
    If the data file is a csv file, columns have to be separated by ';'. 
    The file's columns have to be named as::
    
        X; Y; long; lat; weight; address 
    
    where the data belonging to
    
       * X,Y       : are floats that give the Euclidean coordinates
       * long, lat : are floats giving longitude and latitude
       * weight    : are floats giving the positive weight of the points
       * address   : are strings in quotes that give the address names
    
    Not all of the above data need to be given in the file:
    
    1. It is possible to just give address names. Then geopy is
       used to find longitudes and latitudes, which are then
       converted to Euclidean coordinates.
    2. If the address names are not given, then at least the
       Euclidean coordinates (X,Y) or the geographic coordinates
       (long, lat) need to be present. 
    3. If no weight data are given, all weights are assumed to equal 1

    If the file's extension is ".tsp", the data file is expected to be
    a tsp file. In that case, it need to be a file from the TSP library
    that also shows coordinate information. 

    Parameters
    ----------
    data_file : string
        path and file name of the data file
    
    Returns
    -------
    Y : mx2 numpy array of float
        Y[i] contains Euclidean coordinates of the i-th customer point,
        i=0,...,m-1
    w : numpy array of int or float
        weights of the m customer points
    names : list of string
        name/address of each customer point
            
    Remark
    ------
    If available or generated from address data, the longitude/latitude data and
    address names are stored in the global variables __lo, __la, __names
    """

    lo = None
    la = None
    Y = None
    w = None
    names = None
    
    # Return None if file cannot be found
    if not os.path.isfile(data_file): return Y, w, names

    is_tsp_file = os.path.basename(data_file).split('.')[-1].lower() == 'tsp'

    if is_tsp_file:
        try:
            prob = tsp.load_problem( data_file )
        except:
            print('Cannot read tsp file', data_file )
            return Y, w, names
        if (prob.node_coords is None) or (len(prob.node_coords)==0):
            print('TSP file has no coordinate information')
            return Y, w, names, lo, la
        Y = np.array( [ [coord[0], coord[1]] for coord in prob.node_coords.values() ] )
        w = np.ones(prob.dimension)
        return Y, w, names

    try:
        df = pd.read_csv(data_file,sep=';')
    except:
        raise Exception('Cannot read data file '+data_file)

    header = list(df.keys() )
    has_address = 'address' in header
    has_lola = 'long' in header and 'lat' in header
    has_XY = 'X' in header and 'Y' in header
    has_w = 'weight' in header

    if not has_XY and not has_lola and not has_address:
        print('No coordinates or address data present')
        return Y, w, names

    if has_address: names = list(df.get('address'))

    m = df.shape[0]
    w = np.array(df.get('weight')) if has_w else np.ones(m)
   
    if has_XY: Y = np.column_stack( (list(df.get('X')), list(df.get('Y'))) )
    if has_lola: lo, la = list(df.get('long')), list(df.get('lat')) 
    if Y is None and not has_lola:    
        # Try to get geo-coordinates from geopy
        print('Trying to get geo-coordinates from geopy. This can take a while.')
        geolocator = Nominatim(user_agent="get_long_latt")
        try:
            locations = [ geolocator.geocode( city ) for city in names ]
        except:
            print('No success. Maybe time out happened.')
            return Y, w, names
        lo = [ loc.longitude for loc in locations ]
        la = [ loc.latitude for loc in locations ]        
      
    if has_lola: origin = (min(lo),min(la))
    if Y is None: Y = np.array([lla2xy( lx,ly, origin) for lx,ly in zip(lo,la)] )
            
    global _names, _lo, _la
    _names, _lo, _la = names, lo, la 

    return Y, w, names
       
#----------------------------------------------------------------------

def return_lola(with_names=False):
    """
    Return longitude and latitude data of the points read in. If
    with_names is True, also the address names are returned.
    """
    if with_names: return _lo, _la, _names
    return _lo,_la
    
