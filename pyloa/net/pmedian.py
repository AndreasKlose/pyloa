"""
    Module net.pmedian of package pyloa:
    Methods for solving the p-median problem.
"""

import numpy as np
from itertools import product
from pyloa.net.netprob import NetProblem
from pyloa.mip.model import Model

#--------------------------------------------------------------------------

class PMedian( NetProblem ):
    """
    Class implementing a couple of methods for solving the p-median problem
    on a graph.
    """
    
    def __init__( self, fname=None, dstmat='matrix', orlib=False, d=None, 
                  dmax=None, m=0, n=0, w=None, p=None ):
        """
        Creates an instance of the class PMedian.
        
        Parameters
        ----------   
        see base class NetProblem in module pyloa.net_prob for a description
        of the parameters.
        
        Remark: If the greedy add or subgradient procedure is used, 
            then it is recommended to let d be a numpy (mxn)-array. The 
            procedures then rely on numpy operations and run many times 
            faster. In case that the fname is not None, the same is achieved
            by choosing dstmat='matrix'.    
        """
        super().__init__(fname=fname, dstmat=dstmat, orlib=orlib, d=d, m=m, n=n, w=w, p=p)

        self.__alpha = 2.0 
        """Step size parameter used in the subgradient procedure"""
        
        self.__half = 5
        """Parameter determining when to halve step size parameter alpha"""
        
        self.__miter = 1 
        """Determines number of subgradient steps to equal m+n times this factor"""
        
        self.__epsi = 1.0E-04 
        """Subgradient method stops when step size parameter falls below this value or
        if relative gap between upper and lower bound is smalle than this value."""
           
    #---------------------------------------------
    
    def mipModel( self, x_continuous = True, weakModel = False ):
        """
        Build the MIP model of the p-median problem.
        
        Parameters
        ----------
        x_continuous : bool
            If True (the default), the allocation variables are declared
            as continuous variables, otherwise as binary.
        weakModel : bool 
            If True, the weak model formulation is used, otherwise
            (the default) the strong formulation.
        """
        
        M = Model('p-median problem')
        m, n, p = self._m, self._n, self._p  
    
        # Assignment variables
        x = M.addVarMatrix( m, n) if x_continuous else M.addVarMatrix( m, n, vtype='B' )
    
        # Binary variables equal to 1 if node i is a median node
        y = M.addVars( n, vtype='B' )
    
        # Assignment constraints    
        M.addConstraints( M.sum(x[i,j] for j in range(n))==1 for i in range(m) )
                           
        # No assignment to a non-median node
        if weakModel: 
            M.addConstraints( M.sum(x[i,j] for i in range(m)) <= (m-p+1)*y[j] for j in range(n) ) 
        else:    
            M.addConstraints( x[i,j]-y[j] <= 0 for i,j in product(range(m), range(n)) )
    
        # Number of median nodes
        M.addConstraint( M.sum(y) == p )
    
        # Objective: minimize sum of distances to representative
        M.minimize( M.sum( self._w[i]*self._d(i,j)*x[i,j] for i,j in x.keys() ) ) 
        
        # Remember the the model and the variables
        self.model = M 
        self.x = x 
        self.y = y
        
    #---------------------------------------------
    
    def mipSolve( self, save_facilities=True, save_assignment=True ):
        """
        Solves the MIP formulation of p-median problem using a MIP solver.
        
        Parameters
        ----------
        save_facilities : bool 
            If True the set of facilities open in the obtained
            solution is remembered in the class field self.facilities 
        save_assignment : bool 
            If True the assignment of customers to the open facilities
            in the obtained solution is remembered in the class field 
            self.assigned. 
        """
        M = self.model 
        if not self.model is None: 
            # Initialize running time performance measures
            self._set_starttime( )
                                 
            # Solve model
            M.log_output = not self.silent
            if M.optim( ):
                self._set_comptime( mip_time=True )
                self.wdist = M.ObjVal 
                self.bound = M.ObjBound
                if save_facilities: 
                    self.facilities = list(M.get_solution(self.y, keep_zeros=False, precision=0.1).keys())
                if save_assignment:
                    pairs = M.get_solution( self.x, keep_zeros=False, precision=0.1 ).keys()
                    self.assigned = [0]*len(pairs)
                    for i,j in pairs: self.assigned[i] = j
            
    #---------------------------------------------

    def __addProc( self ):
        """
        Add heuristic for the p-median problem.
        """
        self._set_starttime( )
        talk = not self.silent 
        if talk:
            print('-'*30)
            print('Iter   Objective value')
            print('-'*30)
    
        m, n, p, w = self._m, self._n, self._p, self._w 
        S = list(range(n))
        
        if self._is_matrix:
            # Distances stored as numpy (mxn) array!
            eta = self._max_wdist.astype(float)
            for num in range(p):
                j=np.argmax( np.maximum(0,np.vstack(eta) - np.vstack(w)*self._dmat[:,S[num:]]).sum(axis=0))+num
                S[num], S[j] = S[j], S[num]
                eta = np.minimum( eta, w*self._dmat[:,S[num]] )
                if talk:
                    objv = eta.sum()
                    print('{0:4d}   {1:15.2f}'.format(num+1,objv) )
            self.assigned = self.get_assigned(S[:p])
        else: 
            eta = self._max_wdist + 1.0
            ass = np.zeros(m,dtype=int)
            for num in range(p):
                # Find best facility to add
                j = max(range(num,n), key = lambda j : sum(max(0, eta[i]-w[i]*self._d(i,S[j])) for i in range(m)) )
                S[num], S[j] = S[j], S[num]
                reass = list( filter( lambda i : w[i]*self._d(i,j) < eta[i], range(m) ) )
                ass[reass] = j
                eta[reass] = tuple( map( lambda i : w[i]*self._d(i,j), reass ) )   
                if talk:
                    objv = eta.sum()
                    print('{0:4d}   {1:15.2f}'.format(num+1,objv) )
            self.assigned = ass
                
        if not talk: objv = eta.sum()
        self.wdist = objv 
        self.facilities = S[:p]
        self._set_comptime( )
        self.itr = self._p 
            
    #---------------------------------------------

    def __solveLRsub_no_matrix( self, dualv ):
        """
        Solve Lagrangian subproblem arising when dualizing assignment constraints
        in the p-median problem with Lagrangian multipliers dualv. 
        Distances are assumed to be stored as dictionary or vector 
        """
        m, n, p, w, d = self._m, self._n, self._p, self._w, self._d  
        rho = np.fromiter((sum(min(0, w[i]*d(i,j) - dualv[i]) for i in range(m)) \
                        for j in range(n)),dtype=float) 
        S = np.argsort(rho)[:p]
        cur_bnd = dualv.sum() + rho[S].sum()
        return cur_bnd, S      

    #---------------------------------------------
    
    def __solveLRsub_matrix( self, dualv ):
        """
        Solves Lagrangian subproblem arising when dualizing assignment constraints
        in the p-median problem with Lagrangian multipliers dualv. 
        Assumption is that the distances are provided by a numpy matrix 
        """
        rho = np.minimum(0, np.vstack(self._w)*self._dmat - np.vstack(dualv) ).sum(axis=0)
        S = np.argsort(rho)[:self._p]
        cur_bnd = dualv.sum() + rho[S].sum()
        return cur_bnd, S      
    
    #---------------------------------------------

    def __LR_subgr( self ):
        """ 
        Lagrangian relaxation and heuristic for the p-median problem.
        """        
        m, n, p, w = self._m, self._n, self._p, self._w 
        talk = not self.silent
        if talk:
            print('-'*70)
            print('Iter   Lower bound   Upper bound   Best lower bound   Best upper bound')
            print('-'*70)
    
        self._set_starttime( )
        
        # (1) Function to determine sub-/super-gradient
        # (2) lower bounds on Lagrangian multipliers
        # (3) Function to solve Lagrangian subproblem 
        # (4) Upper bounds on Lagrangian multipliers
        if self._is_matrix:
            get_gradient = lambda dualv,S : 1-(np.vstack(w)*self._dmat[:,S] < np.vstack(dualv)).sum(axis=1)
            min_pos = self._dmat.argmin(axis=1,keepdims=True)
            min_d = np.take_along_axis(self._dmat,min_pos,axis=1)
            np.put_along_axis(self._dmat,min_pos,self._dmat.max()+1,axis=1)
            minLam = w * self._dmat.min( axis=1 )
            np.put_along_axis(self._dmat,min_pos,min_d,axis=1)
            solveLRsub = self.__solveLRsub_matrix
        else:    
            # This case assumes a symmetric matrix of distances between customer and facility nodes
            grad_i = lambda i, S, dvi : 1-sum(map(lambda j : w[i]*self._d(i,j) < dvi, S))
            get_gradient = lambda dualv, S : np.fromiter( (grad_i(i,S,dualv[i]) for i in range(m)),\
                                                           dtype=float)
            if m==n:
                minLam = np.fromiter((w[i]*min(self._d(i,j) for j in range(n) if j!=i ) \
                                      for i in range(m)),dtype=self._dst_type ) 
            else: 
                min_pos = list( map(lambda i : min(range(n), key=lambda j : self._d(i,j)), range(m)) )
                minLam = np.fromiter((w[i]*min(self._d(i,j) for j in range(n) if j!=min_pos[i]) \
                                      for i in range(m)), dtype=self._dst_type)
            solveLRsub = self.__solveLRsub_no_matrix
        maxLam = self._max_wdist
        
        # Initialize Lagrangian multipliers
        dualv = np.divide( p*minLam + (n-p)*maxLam, n)

        # Initialize step size parameters
        alpha = self.__alpha
        maxIter = max(30, self.__miter*(m+n) )
        H = max( self.__half, int( 0.03*maxIter ) )

        # Initial feasible primal solution
        if self.wdist is None:
            self.silent, shut_up = True, self.silent  
            self.__addProc( )
            self.silent = shut_up 
        primal_improve = False
 
        # Perform subgradient steps
        nfail, itr, lobnd = 0, 0, 0.0
        for itr in range(maxIter):    
            # Solve the Lagrangian suproblem
            cur_bnd, S = solveLRsub( dualv )
            # Update best lower bound found so far
            if ( cur_bnd > lobnd ):
                nfail = 0
                lobnd = cur_bnd
            else:
                nfail += 1
                if ( nfail > H ):
                    nfail = 0
                    alpha *= 0.5
                    if alpha < self.__epsi: break      
            # Compute upper bound 
            U = sum(self.min_wdist(S))
            if ( U < self.wdist  ):
                self.wdist = U
                self.facilities = S.copy()    
                primal_improve = True
            # Show progress
            if talk:
                print('{0:4d}   {1:11.2f}   {2:11.2f}   {3:16.2f}   {4:16.2f}'.format(itr,cur_bnd,U,lobnd,self.wdist) )
            # Stop if epsilon-optimal solution reached
            if self.wdist/max(1.0,lobnd) - 1.0 < self.__epsi: break
            # Obtain the subgradient
            g = get_gradient( dualv, S ) 
            # Compute step size
            theta = alpha*( self.wdist - cur_bnd )/np.inner(g,g)
            # Apply subgradient step
            dualv = np.minimum( np.maximum( dualv + theta*g, minLam ), maxLam )

        if talk: print('-'*70)
        if primal_improve: self.assigned = self.get_assigned( self.facilities ) 
            
        self._set_comptime( )
        self.itr = itr 
        self.bound = lobnd 
    
    #---------------------------------------------
    
    def pmp_solve(self, method='weak_MIP', keep_model=False ):
        """
        Computes a solution to the p-median problem.
        
        Parameters
        ----------
        method : str 
            Determines the solution method to be employed:
            
            method = 'weakMIP'
                The weak problem formulation is solved using Cplex.
                Allocation variables are declared as continuous.
            
            method = 'strongMIP'
                The strong problem formulation is solved using Cplex.
                Allocation variables are declared as continuous.
                
            method = 'weakIP'
                The weak problem formulation is solved using Cplex.
                Allocation variables are declared as binary.
            
            method = 'strongIP'
                The strong problem formulation is solved using Cplex.
                Allocation variables are declared as binary.
                
            method = 'greedy'
                A heuristic solution is obtained using the greedy
                add procedure.
                
            method = 'subgradient'
                A heuristic solution and a lower bound is computed
                using Lagrangian relaxation and subgradient optimization.
        
        keep_model : bool
            If True, the docplex model is kept for later purposes,
            otherwise destroyed after the computations. This option
            only applies if docplex is used as solver.
        """ 
        meth = method.lower()
        if 'greedy' in meth:
            self.__addProc()
        elif 'subg' in meth:
            self.__LR_subgr()
        else:
            self.mipModel(x_continuous='mip' in meth, weakModel='weak' in meth) 
            self.mipSolve()
            if not keep_model: self.model.end()
            
    #---------------------------------------------
        
    def default_subgr_params(self):
        """
        Sets the subgradient parameters to default values:
        alpha=2, half=5 and itr=1
        """
        self.__alpha, self.__half, self.__miter = 2.0, 5, 1
     
    #---------------------------------------------
 
    @property 
    def alpha(self):
        """Step length parameter alpha in subgradient procedure"""
        return self.__alpha 
    
    @alpha.setter
    def alpha(self, value): self.__alpha = min(max(0.1, value),2.0) 
    
    @property 
    def half(self):
        """Step size parameter H. The parameter alpha will be halved 
           after H iterations if this is larger than the number of 
           3% of the maximal number of iterations."""
        return self.__half 
    
    @half.setter 
    def half(self, value ): self.__half = max(1,value)
    
    @property 
    def sg_iter(self):
        """Fixes the number of subgradient iterations not to exceed
           miter times (m+n)"""
        return self.__miter 
    
    @sg_iter.setter 
    def sg_iter(self,value): self.__miter = max(1,value)
    
    
