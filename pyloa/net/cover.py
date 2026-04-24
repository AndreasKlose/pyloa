"""
    Module net.cover of package pyloa:
    Methods for solving covering location problems.
"""
import numpy as np
from pyloa.mip.model import Model
from pyloa.net.netprob import NetProblem
from pyloa.net.pmedian import PMedian

#------------------------------------------------------------------------------

class MaxCover( NetProblem ):

    def __init__( self, fname=None, dstmat='matrix', orlib=False, d=None, 
                  dmax=None, m=0, n=0, w=None, p=None ):
        """
        Creates an instance of the class MaxCover.
        
        Parameters
        ----------   
        See base class NetProblem in module pyloa.net_prob for a description
        of the parameters.
        
        Remark: If the greedy add rocedure is used, it is recommended to let d be 
            a numpy (mxn)-array. The procedure then relies on numpy operations and 
            runs many times faster. In case that the fname is not None, the same 
            is achieved by choosing dstmat='matrix'.    
        """
        super().__init__(fname=fname, dstmat=dstmat, orlib=orlib, d=d, m=m, n=n, w=w, p=p)
                       
        self.coverage = 0 
        """Objective function value of any solution to the maximal covering location problem"""
        
        self.__z_continuous = True  
        """If True, variables z_i in the mathematical programming formulation 
        of the maximal covering problem will be declared as continuous variables."""
                       
    #---------------------------------------------
            
    def __open_all( self ):
        """
        Determines solution to maximal covering if all facility nodes are opened.
        """
        self.facilities = list( range(self._n) )
        self.assigned = self.get_assigned( self.facilities )
        self.coverage = sum( self._w ) 
        self.wdist = sum ( self.min_wdist(self.facilities ) )
    
    #---------------------------------------------
    
    def __set_solution(self, objv, S ):
        """
        Register the solution to the maximal covering 
        problem where S is list of open facilities
        and objv the covered demand.
        """
        self.coverage = objv 
        self.facilities = S 
        self.assigned = self.get_assigned( S )
        self.wdist = self.wdist = sum ( self.min_wdist(self.facilities ) )
        
    #---------------------------------------------
    
    def __addProc( self ):
        """
        Add/greedy heuristic for the maximal covering location problem.
        In each iteration, open that facility covering the largest number
        of yet uncovered demand.
    """
        if not self.silent:
            print('-'*30)
            print('Iter   Objective value')
            print('-'*30)
    
        m, n, p, w, d, dmax = self._m, self._n, self._p, self._w, self._d, self._dmax
    
        self._set_starttime()
        if p >= n: 
            self.__open_all()
        elif self._is_matrix:
            S = list(range(n))
            U = np.arange(m,dtype=int)
            objv = 0
            for num in range(p):
                closed = S[num:]
                dem_covered = np.dot(w[U], self._dmat[U][:,closed] <= np.vstack(self._dmax[U]) )
                j = np.argmax(dem_covered)
                objv += dem_covered[j]
                j += num
                U = U[ self._dmat[U,S[j]] > self._dmax[U] ]
                S[num], S[j] = S[j], S[num]
                if not self.silent: print('{0:4d}   {1:15.2f}'.format(num,objv) )
                if len(U)==0: break
            
        else:    
            S = list(range(n)) # ordered set of facilities
            U = list(range(m)) # list of uncovered nodes
            objv = 0
            for num in range(p):
                closed = S[num:]
                dem_covered = list(map(lambda j : sum(w[i] for i in U if d(i,j) <= dmax[i]),closed)) 
                j = np.argmax(dem_covered)
                objv += dem_covered[j]
                j += num
                U = list( filter( lambda i : d(i,S[j]) > dmax[i], U ) )
                S[num], S[j] = S[j], S[num]
                if not self.silent: print('{0:4d}   {1:15.2f}'.format(num,objv) )
                if len(U)==0: break
        
        self.__set_solution(objv, S[:num+1])
        self._set_comptime()
        
    #------------------------------------------------
    
    def __MIPmaxcover( self, keep_model = False ):
        """
        Solves the maximal covering problem using a MIP solver. 
        """
        m, n, p, w, d, dmax = self._m, self._n, self._p, self._w, self._d, self._dmax
        if p >= n: return self._open_all()
        if p==1: return self.__addProc( )
        
        # Build the MIP model    
        M = Model('Maximal covering location problem')
        self.model = M  
    
        # Assignment variables (can be continuous, as they are integer in optimum)
        z = M.addVars( m, ub=1 ) if self.__z_continuous else M.addVars(m, vtype='B')
    
        # Binary variables equal to 1 if node i is a facility
        y = M.addVars(n, vtype = 'B')
    
        # Covering constraints
        M.addConstraints( M.sum( y[j] for j in range(n) if d(i,j) <= dmax[i]+1.0E-5 ) >=z[i] \
                            for i in range(m) )
                             
        # Number of open facilities
        M.addConstraint( M.sum(y) == p )
    
        # Objective: maximize total demand covered
        M.maximize( M.sum( w[i]*z[i] for i in range(m) ) )
             
        M.log_output = not self.silent                    
        # Solve model
        self._set_starttime( )
        if M.optim(  ):
            self.bound = M.ObjBound
            self.__set_solution(M.ObjVal,list(M.get_solution(y,keep_zeros=False,precision=0.1).keys()))
            self._set_comptime(mip_time=True)
        
        if keep_model:
            self.z = z 
            self.y = y
        else:      
            self.model.end() 
          
    #---------------------------------------------
    
    def __mcover_as_pmedian( self, method='mip' ):
        """
        Solves the maximal covering location problem as a p-median
        problem using the indicated method, which can be using
        a MIP solver or the subgradient-based Lagrangian heuristic.
        """  
        m, n = self._m, self._n    
        if self._is_matrix:
            dmat = (self._dmat > np.vstack(self._dmax)).astype(int)
        else:    
            dmat = np.array([list(map(lambda j : int(self._d(i,j)> self._dmax[0]), range(n))) for i in range(m)] )
       
        pmp = PMedian( d=dmat, w=self._w, p=self._p )  
        if method=='mip':
            pmp.pmp_solve( method='strongMIP' )
            self.mip_time = pmp.mip_time 
            self.mip_work = pmp.mip_work 
            self.nodesCount = pmp.nodeCount
        else: 
            pmp.sg_iter = 3
            pmp.pmp_solve(method='subgradient' )
    
        self.itr = pmp.itr 
        self.ctime = pmp.ctime 
        wsum = sum(self._w)
        self.bound = wsum-pmp.bound 
        self.__set_solution(wsum - pmp.wdist,pmp.facilities ) 
         
    #---------------------------------------------

    def mcl_solve(self, method='mip', keep_model=False ):
        """
        Computes a solution to the maximal covering problem.
        
        Parameters
        ----------
        method : str 
            Determines the solution method to be employed:
            
            method = 'IP'
                Formulates the problem as an integer linear
                program and solves it using a MIP solver.
                
            method = 'MIP'
                Formulates the problem as a mixed integer
                linear program and solves it using a MIP solver.     
                    
            method = 'pmp-mip'
                Reformulates the problem as a p-median problem and
                solves this by means of a MIP solver using the strong
                MIP formulation.
                
            method = 'pmp-subgrad'
                Reformulates the problem as a p-median problem
                and applies the subgradient-based Lagrangian
                heuristic to obtain a solution.
                            
            method = 'greedy'
                A heuristic solution is obtained using the greedy
                add procedure.
                          
        keep_model : bool
            If True, the MIP model is kept for later purposes,
            otherwise destroyed after the computations. This option
            only applies if method='IP' or 'MIP'.      
        """
        meth = method.lower()
        if 'greedy' in meth:
            self.__addProc()
        elif 'pmp' in meth:
            if 'sub' in meth:
                self.__mcover_as_pmedian(method='subgradient')
            else: 
                self.__mcover_as_pmedian(method='mip')    
        elif 'ip' in meth: 
            self.__z_continuous = 'mip' in meth 
            self.__MIPmaxcover(keep_model=keep_model)
