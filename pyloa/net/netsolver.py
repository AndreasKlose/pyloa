"""
    Module net.netsolve:
    Implements a class that inherits the solver methods
    from classes MaxCover, PCenter and PMedian.
"""
from pyloa.net.pmedian import PMedian 
from pyloa.net.cover import MaxCover 
from pyloa.net.pcenter import PCenter 

class NetSolver(PMedian,MaxCover,PCenter):
    
    def __init__( self, fname=None, dstmat='matrix', orlib=False, d=None, 
              dmax=None, m=0, n=0, w=None, p=None ):
        """
        Creates an instance of the class NetSolver.
        
        Parameters
        ----------   
        see base class NetProblem in module pyloa.net_prob for a description
        of the parameters.
        """
        super().__init__(fname=fname, dstmat=dstmat, orlib=orlib, d=d, dmax=dmax,\
                         m=m, n=n, w=w, p=p)
        

