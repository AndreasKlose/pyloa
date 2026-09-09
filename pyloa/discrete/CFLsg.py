"""
    Python interface to CFLsg, a C-library of a branch-and-bound algorithm 
    based on Lagrangian relaxation and subgradient optimization for solving 
    the CFLP.    
"""
import numpy as np
import ctypes, pathlib, platform
import os, site, sys

__CFL_Lib = None
__CPX_Lib = None

__int = ctypes.c_int
__double = ctypes.c_double
__bool = ctypes.c_bool
__char = ctypes.c_char
__p_void = ctypes.c_void_p
__p_int = ctypes.POINTER( __int )
__p_double = ctypes.POINTER( __double )
__p_bool = ctypes.POINTER( __bool )
__p_char = ctypes.POINTER( __char )

#---------------------------------------------------------------------

def __loadCFLsg( with_cplex = False ):
    """ 
    Find and load cplex and the CFLsg library.
    
    Parameters
    ----------
    with_cplex : bool
        If False, then libCFLsg.so is assumed to be independent of Cplex,
        which is the case if this library was linked with MCF for solving
        transportation problems instead of using Cplex to accomplish this
        task. Otherwise, with_cplex need to be True.
    """    
    cplexLib = None
    CFLsgLib = None

    onWindows = 'Windows' in platform.system()
    onMac = 'darwin' in platform.system().lower()
    if onWindows:
        cplexDLL = 'cplex*.dll' if with_cplex else None
        CFLsgDLL = 'CFLsg.dll'
    else:
        cplexDLL = 'cplex/**/py*_cplex*.so'
        CFLsgDLL = 'libCFLsg.so'
   
    # Check if the CPLEX callable library is present. On Linux and Mac, we
    # will also have to explicitly load this library before CFLsg is
    # is loaded, as CFLsg depends on Cplex. 
    if with_cplex:
        if onWindows:
            try:
                cpxEnv = next( p for p in os.environ if 'CPLEX_STUDIO_BINARIES' in p )
            except:
                cpxEnv = 'PATH'
            cpxBins = os.getenv(cpxEnv)
            dirLst = [ p for p in cpxBins.split(';') if 'cplex' in p.lower()]            
        else:    
            usrSitePack = next( p for p in sys.path if 'site-packages' in p)
            dirLst = site.getsitepackages()
            dirLst.append( usrSitePack )
        for direc in dirLst:
            for pathName in pathlib.Path( direc ).rglob( cplexDLL ):
                cplexLib = str(pathlib.Path.resolve(pathName))
                if not cplexLib is None:
                    if not onWindows: break
                    if pathlib.PurePath(cplexLib).stem[-1].isnumeric(): break
            if not cplexLib is None: break    

        if cplexLib is None:
            print("Could not find CPLEX dynamic link library.")
            return( False )  
        else:
            print("Cplex loaded. Library is",cplexLib) 

    # The dynamic link library libCFLsg.so (libCFLsg.dll on Windows)
    # is expected to reside in the same directory as this module.
    print("Searching for",CFLsgDLL)
    if onWindows:
        prefix = 'win//'
    elif onMac:
        prefix = 'mac/arm64/' if 'arm64' in platform.platform() else 'mac/x86_64/'
    else:
        prefix = 'linux/'
    CFLsgHome=pathlib.Path(__file__).resolve().parent
    CFLsgFile = pathlib.PurePath.joinpath(CFLsgHome,prefix+CFLsgDLL)
    if CFLsgFile.is_file():
        CFLsgLib = str( CFLsgFile )
    else:
        print("Could not find CFLsg dynamic link library.")
        return( False )

    # Libraries found. Now try to load them. 
    # If Cplex need to be used, the load Cplex first.
    global __CPX_Lib
    global __CFL_Lib
    if not onWindows:
        if with_cplex:
            __CPX_Lib = ctypes.CDLL(cplexLib, mode=ctypes.RTLD_GLOBAL)
            if __CPX_Lib is None: return( False )
        __CFL_Lib = ctypes.CDLL(CFLsgLib, mode=ctypes.RTLD_GLOBAL)
    else: 
        __CFL_Lib = ctypes.CDLL(CFLsgLib, winmode=0)
    if __CFL_Lib is None:
        print("Error occured when loading library ",CFLsgLib )
        return( False )
    else:
        print("Library",CFLsgLib,"loaded.")    

    return( True )
    
#---------------------------------------------------------------------

def solveCFLP( d, s, f, c, nodeLim=0, screenOn=1 ):
    """
    Invokes the CFLsg solver for the CFLP.

    Parameters
    ----------
    d : numpy array of int 
        customer demands
    s : numpy array of int 
        facility capacities
    f : numpy array of float
        fixed facility costs
    c : numpy 2d-array of float 
        c[j][i] is the total cost of supplying all of customer
        i's demand from facility j, where 0<=i<m and 0<=j<n.    
    nodeLim: int (optional)
        limit on number of nodes (if 0, no limit is set)
    screenOn: int (optional) 
        if 1, the solver sends information to the screen   
    
    Returns
    -------
    status : int
        status of the result     
    UB : float
        best upper bound found in the search
    OPN: list of int
        set of open facilities
    TIM: float
        CPU time spent               
    """
    if __CFL_Lib is None:
        print("CFLsg Library not loaded!")
        return 0.0, 0.0, [], 0.0

    nn, mm = c.shape
    
    n = __int( nn )
    m = __int( mm )
    env = __p_void(None)
    flow = __bool(0)

    ndeLim = __int( nodeLim )
    doDisplay = __bool( screenOn )
    
    __CFL_Lib.CFLsetScreen( doDisplay )
    __CFL_Lib.CFLsetNodeLim( ndeLim );
    
    dd = d.astype(__int) if not d is None else np.ones( mm, dtype=__int )
    ss = s.astype(__int) if not s is None else np.full( nn, dd.sum(), dtype=__int )
    ff = f.astype(__double, copy=False )
    pff = ff.ctypes.data_as( __p_double ) 
    pss = ss.ctypes.data_as( __p_int )
    pdd = dd.ctypes.data_as( __p_int ) 

    cc = c.astype(__double, copy=False )
    pcc = (cc.__array_interface__['data'][0] 
      + np.arange(cc.shape[0])*cc.strides[0]).astype(np.uintp) 

    y = np.zeros( nn, dtype=__bool )
    py = y.ctypes.data_as( __p_bool )
    objv = __double( 0.0 )
    
    # Call CFLsg solver 
    __pp_double = np.ctypeslib.ndpointer( dtype=np.uintp, ndim=1, flags='C' ) 
    __CFL_Lib.CFLoptim.argtypes = [ __p_void, __int, __int, __bool, __p_int, __p_int,\
                                    __p_double, __pp_double, __p_double, __p_bool]
    __CFL_Lib.CFLoptim.restype = __int
    status = __CFL_Lib.CFLoptim( env, m, n, flow, pss, pdd, pff, pcc, ctypes.byref(objv), py )
    __CFL_Lib.CFLgetTtim.restype = __double
    TIM = __CFL_Lib.CFLgetTtim()
    OPN = list( filter( lambda j : y[j], range(nn) ) )
    return status, objv.value, OPN, TIM     
    
#---------------------------------------------------------------------

is_ready = __loadCFLsg()

