"""
    Read data of a discrete facility location problem.
    
    Repositories with problem instances are in particular:
      
    - For the CFLP: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/  
    - For the UFLP: https://resources.mpi-inf.mpg.de/departments/d1/projects/benchmarks/UflLib/data-format.html
"""
import numpy as np
import os
from urllib.request import urlretrieve
from itertools import product

ORLib = 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/'

#---------------------------------------------------------------------

def __readAvellaOld( dataFile ):
    """
    Read old Avella-Boccia CFLP instance from file "dataFile".

    Parameters
    ----------
    dataFile : str
        Name of the data file

    Returns
    -------                         
    d : numpy array of int
        customer demands
    s : numpy array of int
        facility capacities 
    f : numpy array of float
        fixed facility cost
    c : numpy 2d-array of float
        c[j][i] is the UNIT cost of supplying customer i from facility j. 
    """
    F = open( dataFile, "r" )
    data = F.read().split()
    F.close()
    nCust = int(data[0])
    nFac = int(data[1])
    d = np.array( data[2:nCust+2], dtype=int )
    s = np.array( data[nCust+2:nCust+nFac+2], dtype=int )
    f = np.array( data[nCust+nFac+2:nCust+2*nFac+2], dtype=float )
    c = np.array(data[nCust+2*nFac+2:],dtype=float).reshape(nFac,nCust)
   
    return d, s, f, c

#---------------------------------------------------------------------

def __readAvellaNew( dataFile, unitCost=True ):
    """
    Read new Avella-Boccia CFLP instance from file dataFile. 
    Returns exactly the same as __readAvellaOld.
    """
    F = open( dataFile, "r" )
    data = F.read()
    F.close()
    data = data.replace('DEMAND:[','')
    data = data.replace('CAP:[','')
    data = data.replace('SETUPCOST:[','')
    data = data.replace('TRANSPCOST:[','')
    data = data.split(']')
    for i in range(len(data)): data[i] = data[i].split()

    # Demand, capacities, fixed costs, unit supply cost matrix
    d = np.array(data[0],dtype=int)
    s = np.array(data[1],dtype=int)
    f = np.array(data[2],dtype=float)
    c = np.array(data[3],dtype=float).reshape(len(s),len(d))
        
    return d, s, f, c

#---------------------------------------------------------------------

def __readGuastaroba( dataFile, unitCost=True, scale=1 ):
    """
    Reads a Guastaroba CFLP instance from file dataFile. Returns
    are the same as above. In case of these instances demand
    and capacity data are not necessarily integers. Non-integral
    demand values are thus multiplied by the provided scaling factor
    and rounded down, and supply values are multiplied by scale
    and rounded to the nearest integer. As for these instance
    supply (transportation costs) are costs per unit, also these
    cost data are multiplied by the scaling factor.
    """
    F = open( dataFile, "r" )
    data = F.read().split()
    F.close()

    nFac = int(data[0])
    nCust = int(data[1])

    # Gustaroba stores demands and capacities as floats.
    df = np.array( data[2:nCust+2], dtype=float)
    sf = np.array( data[nCust+2:nCust+nFac+2], dtype=float)
    f = np.array( data[nCust+nFac+2:nCust+2*nFac+2], dtype=float)
    c = np.array(data[nCust+2*nFac+2:],dtype=float).reshape(len(sf),len(df))
    
    non_integral = np.any( df % 1 > 0.0 ) or np.any( sf % 1 > 0.0)
    if non_integral and scale > 1:
        df *= scale 
        sf *= scale 
        c  *= scale
    
    d = np.floor( df ).astype(int)
    s = np.round( sf ).astype(int)
        
    return d, s, f, c
   
#---------------------------------------------------------------------

def __getCustomerData( dataFile ):
    """
    Read customer data from a file following the structure 
    of the Goertz-Klose instances for the CFLP and TSCFLP.
    """
    F = open( dataFile, "r" )
    for line in F:
        if '[CUSTOMERS]' in line.upper(): break
    d = []
    F.readline()
    for line in F:
        if line.strip() == '': break
        d.append(line.split()[0])    
    F.close()
    return np.array(d,dtype=int)

#---------------------------------------------------------------------

def __getFacilityData( dataFile, stage=2, two_level=False ):
    """
    Read data of stage 'stage' facilities from a file following
    the structure of the Goertz-Klose instances for the CFLP and TSCFLP.
    Facilities are depots if stage=2 and plants if stage=1. If
    two-level is False, there are no plant fixed cost (as their location
    is given), otherwise there are plant fixed cost as well.
    """
    F = open( dataFile, "r" )
    sect = '[DEPOTS]' if stage > 1 else '[PLANTS]'
    for line in F:
        if sect in line.upper(): break
    s = []
    f = [] if stage==2 or two_level else None
    F.readline()
    for line in F:
        if line.strip() == '': break
        dataRow = line.split()
        s.append( dataRow[0] ) 
        if not f is None: f.append( dataRow[1] ) 
    F.close()
    if f is None: return np.array(s,dtype=int)
    return np.array(s,dtype=int), np.array(f,dtype=float)

#---------------------------------------------------------------------

def __getCostMatrix( n, m, dataFile, header='[COSTMATRIX]',\
                     alt_header='[COSTMAT]' ):
    """
    Read a cost matrix from a file from  a file structured as the 
    CFLP and TSCFLP Goertz-Klose instances.
    """
    sub_header = '[MATRIX]'
    F = open( dataFile, "r" )
    for line in F:
        l = line.upper()
        if header in l or alt_header in l: break
    for line in F:
        if sub_header in line.upper(): break 
    F.readline()
    c = np.zeros( (n,m),dtype=float )
    for j, line in enumerate(F):
        if line.strip() == '': break
        c[j,:] = np.array([float(l) for l in line.split()],dtype=float )
    F.close()
    return( c )

#---------------------------------------------------------------------

def __readGortzKlose( dataFile ):
    """
    Reads a Goertz-Klose instance from file dataFile. Returns are the same 
    as for __readAvellaOld and __readAvellaNew with the exception of the 
    supply cost matrix c, which is returned with c[j][i] being the cost of 
    supplying all of customer's i demand from facility.
    """            
    d = __getCustomerData( dataFile )
    s, f = __getFacilityData( dataFile )
    c = __getCostMatrix( len(s), len(d), dataFile )
    
    return d, s, f, c

#---------------------------------------------------------------------

def __readORLib( dataFile ):
    """
    Read an ORLIB CFLP data instance
    """
    F = open( dataFile, 'r')
    data = F.read().split()
    F.close()
    n = int(data[0])
    m = int(data[1])
    s = np.array([data[j] for j in range(2,2+2*n) if j%2==0], dtype=int)
    f = np.array([data[j] for j in range(2,2+2*n) if j%2>0], dtype=float)
    D = np.array(data[2+2*n:]).reshape(m,n+1)
    d = D[:,0].astype(int)
    c = np.zeros( (n,m),dtype=float )
    c[:,:] = (D[:,1:].transpose()).astype(float)
    
    return d, s, f, c
    
#---------------------------------------------------------------------

def __read_ufl_simple( dataFile ):
    """
    Read UFLP instance in the simple UflLib format
    """
    F = open( dataFile, 'r')
    data = F.read().split()
    F.close()
    for l, d in enumerate(data):
        if d.isnumeric(): break 
    n = int(data[l])
    m = int(data[l+1])
    D = np.array(data[l+3:]).reshape(n,m+2)
    f = np.array(D[:,1],dtype=float)
    J = np.array(D[:,0],dtype=int)-1
    c = np.array(D[:,2:],dtype=float)
    return f, c[J]         
                 
#---------------------------------------------------------------------

def read_cflp( dataFile, formt='GK', scale=1):
    """
    Reads data of a CFLP instance from a file.

    Parameters
    ----------
    dataFile : str
        Name of the data file
    formt : str
        format of the data file, that is,
            "AO" for the old Avella-Boccia CFLP instances
            "AN" for the new Avella-Boccia CFLP instances
            "G"  for the Guastaroba CFLP instances
            "GK" for the Goertz-Klose CFLP instances (Default)
            "OR" for an ORLIB CFLP instance
    scale : int, optional
        Only used for Guastaroba instances. 
        If scale=1, demands and capacities will be returned as floats;
        otherwise they are multiplied by 'scale' and the non-integral 
        capacities rounded to the nearest integer.                      

    Returns
    -------
    d : numpy array of int or float 
        customer demands
    s : numpy array of int or float
        facility capacities 
    f : numpy array of float
        fixed facility costs
    c : numpy 2d-array of float 
        c[j,i] is either the unit cost or the total cost of
        supplying customer i from a facility at site j.
    """
    formt = formt.upper()      
    if not os.path.isfile( dataFile ): 
        if formt == 'OR':
            #Try to download the instance from the OR library
            try:
                url = ORLib+dataFile
                urlretrieve( url, dataFile )  
            except:
                raise Exception("Cannot download file "+dataFile)
        else:
            raise Exception("Cannot read file "+dataFile )  
                     
    if formt == 'AO':
        d, s, f, c = __readAvellaOld( dataFile )
    elif formt == 'AN':
        d, s, f, c = __readAvellaNew( dataFile )
    elif formt == 'G':
        d, s, f, c = __readGuastaroba( dataFile, scale=scale )
    elif formt == 'GK':
        d, s, f, c = __readGortzKlose( dataFile )
    elif formt == 'OR':
        d, s, f, c = __readORLib( dataFile )
    else:
        return None, None, None, None 
    
    return d, s, f, c

#---------------------------------------------------------------------

def read_uflp( dataFile, formt='OR' ):
    """
    Reads data of a UFLP instance from a file.

    Parameters
    ----------
    dataFile : str
        Name of the data file
    formt : str
        format of the data file, that is,
            "OR"     for data files just the same as the CFLP ORLib files
            "SIMPLE" for data files in UflLib simple format                      

    Returns
    -------
    f : numpy array of float
        fixed facility costs
    c : numpy 2d-array of float 
        c[j,i] is either the unit cost or the total cost of
        supplying customer i from a facility at site j.
    """      
    if not os.path.isfile( dataFile ): 
        raise Exception("Cannot read file "+dataFile)
    if formt.upper() == 'OR':                     
        _, _, f, c = __readORLib( dataFile )
    else: 
        # Simple format is assumed
        f, c = __read_ufl_simple( dataFile )
    return f, c

#-------------------------------------------------------------------

def read_tscflp( dataFile, two_level = False, mcf=False ):
    """
    Read an instance of a 2-stage capacitated facility location problem
    (two_level=False) or of a 2-level CFLP (two_level=True) from a file 
    that uses the format of the test instances used in Klose (1999, 2000).
    If mcf is True, the cost for supplying customers is returned
    as a 3-dimensional numpy array in case of a two-level problem.
    
    Parameters
    ----------
    dataFile : str
        Name (and path) of the data file
    two-level : bool, optional 
        True, if the file contains data for a two-level problem
        so that there are fixed cost for both facilities on
        stage 1 (plants) and on stage 2 (depots). Otherwise,
        fixed cost are only read for stage 2 facilities.
    mcf : bool, optional 
        If True, the as single cost matrix as a numpy 3d-array
        is returned so that c[i,j,k] is the cost to supply all
        of customer k's demand from facility j on stage 2 and
        facility i on stage 1. Note that this only makes sense
        if the problem is a two-level one with location decisions
        to be made on both stages. If mcf is False (the default)
        a tuple of two numpy 2d-arrays is returned. The first
        one gives the unit cost to supply facilities on stage 2
        from facilities on stage 1. The second one shows the
        cost to supply all of a customer's demand from a
        facility on stage 2. Rows of the first matrix correspond
        to stage 1 facilities. Rows of the second matrix to
        stage 2 facilities.
        
    Returns
    -------
    d : numpy array of int 
        Array of customer demands
    s : tuple of numpy arrays of int 
        Array of capacities of stage 1 and stage 2 facilities
    f : numpy array of float or tuple of such arrays
        Fixed cost of stage 2 facilities if two_level=False.
        Otherwise the tuple of fixed costs of facilities
        on stage 1 and stage 2.
    c : tuple of two numpy 2d-arrays or a numpy 3d-array of float
        If mcf=False, c[0][i,j] is the cost to supply 1 unit
        of product from facility i at stage 1 to facility j
        at stage 2, and c[1][j,k] is the cost of supplying
        all of customer k's demand from stage 2 facility j.
        If mcf=True, c[i,j,k] is the cost to supply all
        of customer k's demand from facility j at stage 2
        and facility i at stage 1.
    """
    if not os.path.isfile( dataFile ):
        raise Exception("Cannot read file "+dataFile)
    
    # Get demand data            
    d = __getCustomerData( dataFile )
    # Get depot data (2nd stage facilities) 
    s, f = __getFacilityData( dataFile )
    # Get plant data (1st stage facilities)
    if two_level:
        S, F = __getFacilityData( dataFile, stage=1, two_level=two_level )
    else:
        S = __getFacilityData( dataFile, stage=1 )
    # Get cost matrix (unit cost) on 1st stage 
    # In the file, rows below to depots not to plants. Hence transpose it
    C = __getCostMatrix( len(s), len(S), dataFile, header='[FSCOSTMAT]' ).T
    # Get cost matrix on second stage
    c = __getCostMatrix( len(s), len(d), dataFile, header='[SSCOSTMAT]') 
    
    if not two_level: return d, (S,s), f, (C,c) 
    if not mcf: return d, (S,s), (F,f), (C,c)
    
    # Create 3d cost matrix for multi-commodity formulation
    p, n, m = len(S), len(s), len(d)
    cc = np.zeros( (p,n,m) )
    for k in range(m): cc[:,:,k] = C*d[k]
    for i in range(p): cc[i,:,:]+= c
    return d, (S,s), (F,f), cc

#-------------------------------------------------------------------

def read_tlcflp( dataFile, mcf=False ):
    """
    Read an instance of a 2-level capacitated facility location problem
    that uses the same format as the instances from Fernandes et al
    (2014). That is, the file is structured as follows:
    
    1. 1st line is : number p of plants, n of depots and m of customers
    2. m lines with customer demand
    3. p lines with plant capacity and fixed cost
    4. matrix with supply cost per unit from plants (rows) to depots (columns)
    5. n lines with depot capacity and fixed cost
    6. matrix of supply cost cost per unit from depots (rows) to customers (columns)
    
    If mcf is True, the cost matrix (unit cost!) will be returned as a 3d-matrix
    indicating that the multi-commodity formulation of the problem is to
    be used.
    
    Parameters
    ----------
    dataFile : str 
        Name (and path) of the data file 
    mcf : bool, optional 
        If True, a numpy 3d-array of float is returned as cost matrix.
        Otherwise a tuple of two numpy 2d-arrays is returned.
        
    Returns
    -------
    d : numpy array of int 
        Array of customer demands
    s : tuple of numpy arrays of int 
        Array of capacities of stage 1 and stage 2 facilities
    f : tuple of two numpy array of float
        The tuple of fixed costs of facilities on stage 1 and 2.
    c : tuple of two numpy 2d-arrays or a numpy 3d-array of float
        If mcf=False, c[0][i,j] is the cost to supply 1 unit
        of product from facility i at stage 1 to facility j
        at stage 2, and c[1][j,k] is the cost of supplying
        all of customer k's demand from stage 2 facility j.
        If mcf=True, c[i,j,k] is the cost to supply 1 unit
        of customer k's demand from facility j at stage 2
        and facility i at stage 1.
    """
    if not os.path.isfile( dataFile ):
        raise Exception("Cannot read file "+dataFile)
    
    ff = open( dataFile, 'r' )
    data = (ff.read()).split('\n')
    ff.close()
    
    # Number of plants, satellites, and customers
    (p,n,m) = map(lambda x:int(x),data[0].split())
    
    # customer demands
    d = np.array(data[1:m+1]).astype(int)

    # plant capacities and fixed costs
    SF = np.array(data[m+1:p+m+1])
    S  = np.array([SF[i].split()[0] for i in range(p)]).astype(int)
    F  = np.array([SF[i].split()[1] for i in range(p)]).astype(float)

    # unit transportation costs plants -> depots
    C = np.array([data[p+m+1:2*p+m+1][i].split() for i in range(p)] ).astype(float)
    
    # Depot capacities and fixed costs
    sf = data[2*p+m+1:2*p+m+n+1]
    s = np.array([sf[j].split()[0] for j in range(n)]).astype(int)
    f = np.array([sf[j].split()[1] for j in range(n)]).astype(float)
    
    # Unit transportation costs depots -> customers
    c = np.array([data[2*p+m+n+1:-1][j].split() for j in range(n)] ).astype(float)
    
    if not mcf: return d, (S,s), (F,f), (C,c)
    
    # Create 3d cost matrix for multi-commodity formulation
    p, n, m = len(S), len(s), len(d)
    cc = np.zeros( (p,n,m) )
    for k in range(m): cc[:,:,k] = C
    for i in range(p): cc[i,:,:]+= c
    
    # check cost matrix
    for i,j in product(range(p),range(n)):
        for k in range(m):
            cst = C[i,j] + c[j,k]
            if abs(cst-cc[i,j,k]) > 1.0E-07:
                print('error')
    return d, (S,s), (F,f), cc

#---------------------------------------------------------------------

def store_cflp_GK( fname, d, s, f, c, unitCost=True ):
    """
    Write CFLP data to file fname using GK format.
    
    Parameters
    ----------
    fname : str
        Name of file where to write the data. 
        Warning: Any existing file of the name and path will be overwritten!
    d : numpy array of float or int
        customer demands
    s : numpy array of float or int
        facility capacities
    f : numpy array of float 
        fixed facility costs
    c : numpy 2d-array of float
        c[j,i] is either the cost per unit of supplying
        customer i from facility j (if unitCost=True)
        or the total cost of supplying customer i
        from facility j (if unitCost=False)
    unitCost : bool
        see argument c 
    """
    n, m = c.shape
    F = open(fname,'w')
    F.write('[CFLP-PROBLEMFILE]\n')
    F.write('No comment\n\n')
    F.write('[CUSTOMERS]\n')
    F.write('demand name\n')
    for i in range(m): F.write(str(d[i])+' Customer '+str(i)+'\n')
    F.write('\n')
    F.write('[DEPOTS]\n')
    F.write('capacity fixcost\n')
    for j in range(n): F.write(str(s[j])+' '+str(f[j])+'\n')
    F.write('\n')
    F.write('[COSTMAT]\n')
    F.write('Matrix c(j,i). Rows=Depots\n')
    F.write('[MATRIX]\n')
    F.write('Dim '+str(n)+' '+str(m)+'\n')
    for j in range(n):
        for i in range(m):
            F.write(str(c[j,i])+' ')
        F.write('\n')
    F.close()
