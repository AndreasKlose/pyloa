"""
    Module mip.model of package pyloa
    Used to either interface docplex or gurobipy
"""
from importlib.util import find_spec

# Check if docplex or gurobipy is installed
__cplex_ok = not find_spec('docplex') is None 
__grb_ok = not find_spec('gurobipy') is None

__mipSolver = None 
if __cplex_ok:
    from pyloa.mip.cpx_mip import CPXmodel 
    __mipSolver = 'cplex'
if __grb_ok:
    from pyloa.mip.grb_mip import GRBmodel 
    if __mipSolver is None: __mipSolver = 'gurobi'

#----------------------------------------------

def set_mipSolver( solver=None ):
    """
    Set the MIP solver to be used.
    
    Parameters
    ----------
    solver : str or None 
         If None, the name of the MIP solver is
         returned. Otherwise, the name of the
         solver to used (either 'cplex' or 'gurobi').
    """
    global __mipSolver
    if solver is None: return __mipSolver 
    s = solver.lower()
    if s in 'cplex' and __cplex_ok:
        __mipSolver = 'cplex'
    elif s in 'gurobi' and __grb_ok: 
        __mipSolver = 'gurobi'
    
#----------------------------------------------

def Model( name=None ):
    """
    Create and return an instance of either
    a docplex.mp.model or gurobipy model.
    
    Parameters
    ----------
    name : str or None 
        Name of the model
    """
    if __mipSolver == 'cplex': return CPXmodel(name)
    if __mipSolver == 'gurobi':  return GRBmodel(name)
    return None
