"""
    Module plane.parser of package pyloa:
    Read data of a planar location problem.
"""
import numpy as np
import pandas as pd
import os
import tsplib95 as tsp

from importlib.util import find_spec

from geopy.geocoders import Nominatim
from geopy.point import Point 
from geopy.extra.rate_limiter import RateLimiter

from pyloa.util import lla2xy

_lo = None     # list of longitude data
_la = None     # list of latitude data
_names = None  # list of customer/city names

# The applicable geo-servers
_geo_server = 'Nominatim' # Alternative is OpenCage (https://opencagedata.com/)

# Parameters for Nominatim
_min_delay = 1   # minimal delay in seconds between success requests 
_max_retries = 5 # Number of retries for such requests
_domain = None   # If None, default domain nominatim.openstreetmap.org is used
 
# OpenCage geocoding if present
if not find_spec('opencage') is None:
    from opencage.geocoder import OpenCageGeocode

_api_key = None   # API key for the geo-server   
_geo_coder = None # The geocoding function applied
    
#----------------------------------------------------------------------

def set_geoServer( server = 'Nominatim', api_key=None ):
    """
    Choose the geo-server to be used for geo-coding addresses. Possible 
    choices or Nominatim (nominatim.openstreetmap.org) or OpenCage
    (opencage.com). Note that the latter is commercial. For accessing
    the server you need to have an account at OpenCage and an API-key.
    The API-key can either be provided as an argument to this functioin
    or via the environment variable OPENCAGE_API_KEY. Note that in case 
    you use OpenCage's free service, the number of requests (addresses to 
    geocode) is limited to 2500 per day.
    The default server is Nominatim. Also a locally installed copy
    of Nominatim could be used. In that case, you have to set the
    domain variable in function set_Nominatim_params to your localhost.
    
    Parameters
    ----------
    server : str 
        Name of the server to be used (eiter Nominatim or OpenCage)
    api_key : str, optional
        Your api_key for OpenCage if not stored using environment variable.
    """
    global _geo_server, _api_key 
    
    _geo_server = server 
    if server.lower() == 'opencage':
        _api_key = os.environ.get('OPENCAGE_API_KEY') if api_key is None else api_key
        if _api_key is None: _geo_server = None

#----------------------------------------------------------------------

def set_Nominatim_params( min_delay=1, max_retries=5, domain=None ):
    """
    Change the default values used for the min_delay and the number of
    retries when sending requests to the Nominatim geo-server. For
    going back to default values, simply use set_Nominatim_params().
    
    Parameters
    ----------
    min_delay : int 
        Is the minimum delay in seconds
    max_retries : int
        Is the maximal number of retries.
    """
    global _min_delay, _max_retries, _domain
    
    if min_delay > 0: _min_delay = min_delay 
    if max_retries > 0: _max_retries = max_retries 
    _domain = domain
    
#----------------------------------------------------------------------

def __open_cage_geocode( address ):
    """
    Geocode address using OpenCage's geocoder.
    """
    results = _geo_coder.geocode(address, no_annotations='1')
    lng, lat = results[0]['geometry']['lng'], results[0]['geometry']['lat']
    return Point(latitude=lat, longitude=lng ) 
    
#----------------------------------------------------------------------

def __get_geocoder():
    """
    Return the function for geo-coding addresses.
    """
    global _geo_coder
    
    my_server = _geo_server.lower()
    if my_server == 'nominatim':
        _geo_coder = Nominatim(user_agent='get_long_lat') if _domain is None else \
                     Nominatim(user_agent='get_long_lat', domain=_domain)
        return RateLimiter( _geo_coder.geocode, min_delay_seconds=_min_delay,\
                           max_retries=_max_retries)
    if my_server == 'opencage' and not _api_key is None:
        _geo_coder = OpenCageGeocode( _api_key )
        return __open_cage_geocode
    
    return None

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
        geocode = __get_geocoder()
        if _geo_server is None:
            print('No geo-server available.')
            return Y, w, names 
        # Try to get geo-coordinates from the geo-server
        print('Trying to get geo-coordinates from the geo-server. This can take a while.')
        try:
            locations = [ geocode( city ) for city in names ]
        except:
            print('No success. Maybe time out happened.')
            return Y, w, names
        lo = [ loc.longitude for loc in locations ]
        la = [ loc.latitude for loc in locations ]    
        has_lola = True    
      
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

