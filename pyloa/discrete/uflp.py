"""
    Module discrete.uflp of package pyloa:
    Methods for solving the uncapacitated facility location problem.
"""
import numpy as np
from pyloa.discrete.dflprob import DFLProblem
from itertools import product

#--------------------------------------------------------------------------

class UFLP( DFLProblem ):
    """
    Class implementing methods for solving the uncapacitated facility
    location problem. Recall that class CFLP from discrete.cflp can
    be used to solve instances of the UFLP as well.
    """
    def __init__( self, fname=None, formt='UFL-OR', f=None, c=None, d=None ):
        """
        Creates instance of an UFLP. 
        
        See base class 'DFLProblem' in module dflprob.
        """
        super().__init__(fname=fname, formt=formt, unitCost=not d is None,\
                         f=f, c=c, d=d )
        
        self.itr = 0
        """Iteration counter."""
        
        self.default_subgr_params()
                   
    #---------------------------------------------
    
    def default_subgr_params(self):
        """
        Set subgradient optimization parameters to default values.
        """
        self.__alpha, self.__half, self.__miter, self.__epsi = 2.0, 5, 1, 1.0E-04

    #---------------------------------------------
    
    def __addProc( self ):
        """
        Add heuristic for the UFLP.
        """
        self._set_starttime( )
        if not self.silent:
            print('-'*30)
            print('    ADD PROCEDURE')
            print('Open   Objective value')
            print('-'*30)
    
        c, f = self.c, self.f 
        n, _ = self.c.shape
        S = list()            # List of open facilities
        CL = np.arange(n)     # Array of facilities that might be opened
        eta = c.max( axis=0 ) # Cost of serving customers in current solution
        num = 0               # Number of open facilities
        objv= 0.0             # Current objective value
        for num in range(n):
            # Calculate reduction in cost by including any facility
            cstsav = np.maximum(0, eta - c[CL]).sum(axis=1)-f[CL]
            j = np.argmax( cstsav )
            max_sav = cstsav[j]
            if not max_sav > 0.0: break 
            jj = CL[j]
            S.append(jj) 
            eta = np.minimum( eta, c[jj] )
            objv= eta.sum() + f[jj] if num==0 else objv - max_sav 
            if not self.silent: print('{0:4d}   {1:15.2f}'.format(num+1,objv) )
            # Remove facility j and all those not showing positive saving
            # from the candidate list CL.
            cstsav[j] = 0.0
            CL = CL[np.where (cstsav > 0.0)[0]] 
            if len(CL)==0: break 
      
        self._set_comptime()
        self.cost  = objv
        self.fcost = f[S].sum()
        self.tcost = self.cost - self.fcost
        self.facilities = S 
        self.assigned = list(map(lambda j : S[j], np.argmin(c[S], axis=0)))
        self.itr = num 
        
        if not self.silent: print('-'*30)

    #---------------------------------------------

    def __dropProc(self):
        """
        Drop heuristic for the UFLP
        Still to be implemented
        """ 
        self._set_starttime( )
        if not self.silent:
            print('-'*30)
            print('    DROP PROCEDURE')
            print('Open   Objective value')
            print('-'*30)
    
        c, f = self.c, self.f 
        num, _ = self.c.shape
        
        # List of open facilities and candidates to drop
        CL = np.arange(num, dtype=int ) 
        S = list(CL)
        
        # Cheapest and 2nd cheapest cost to supply each customer
        cmin1, cmin2 = np.partition( c, kth=1, axis=0 )[0:2,:]
        
        # Cost of the solution
        objv = f.sum() + cmin1.sum()
        if not self.silent: print('{0:4d}   {1:15.2f}'.format(num,objv) )
        
        # Close facilities as long as cost can be reduced this way
        while num > 1:
            cstsav = f[CL] - np.maximum(0, cmin2 - c[CL]).sum(axis=1)
            j = np.argmax( cstsav )
            max_sav = cstsav[j]
            if not max_sav > 0.0: break 
            jj = CL[j]
            S.remove(jj)
            objv -= max_sav
            num  -= 1 
            if not self.silent: print('{0:4d}   {1:15.2f}'.format(num,objv) )
            # Remove facility j incl. those not showing positive saving
            # from the candidate list CL.
            cstsav[j] = 0.0
            CL = CL[np.where (cstsav > 0.0)[0]] 
            if len(CL)==0: break 
            # Adjust smallest and second smallest supply cost
            chg_min = np.where( c[jj] <= cmin2 )[0]
            if len(chg_min) > 0:
                cmin2[chg_min] = np.partition( c[S][:,chg_min], kth=1, axis=0)[1,:]
                    
        self._set_comptime()
        self.cost  = objv
        self.fcost = f[S].sum()
        self.tcost = self.cost - self.fcost
        self.facilities = S 
        self.assigned = list(map(lambda j : S[j], np.argmin(c[S], axis=0)))
        self.itr = num 
        
        if not self.silent: print('-'*30)
        
    #---------------------------------------------
    
    def __interchg(self): 
        """
        Interchange procedure to be applied to the 
        solution obtained either by the drop or add 
        procedure.
        """ 
        self._set_starttime( )
        if not self.silent:
            print('-'*30)
            print('INTERCHANGE PROCEDURE')
            print('Iter   Objective value')
            print('-'*30)
    
        c, f, S = self.c, self.f, self.facilities 
        
        # Cheapest and 2nd cheapest cost to supply each customer
        cmin1, cmin2 = np.partition( c[S], kth=1, axis=0 )[0:2,:]
        
        # Cost of the solution
        itr, objv = 0, f[S].sum() + cmin1.sum()
        if not self.silent: print('{0:4d}   {1:15.2f}'.format(itr,objv) )
        
        # Flag indicating if a facility is closed or open
        is_open = np.zeros( len(f), dtype=bool )
        is_open[S] = True
        N_S = np.where( is_open==False)[0]
        
        # Function returning the second smallest allocation cost
        # when facility k is opened but not yet a facility closed.
        sec_min = lambda k: np.maximum( np.minimum(c[k], cmin2), cmin1 )
        
        # Function returning reduction in allocation cost if facility k is opened
        # but not yet any other facility opened.
        c_reduce = lambda k : np.maximum(0, cmin1 - c[k] ).sum()
        
        # Function returning increase in these cost if j is dropped and k opened
        c_increase = lambda j, k : np.maximum(0,sec_min(k)-c[j]).sum()
        
        # Function return the total cost saving if facility j is closed and k opened
        costsav = lambda j,k: f[j] - f[k] + c_reduce(k) - c_increase(j,k)
         
        while True:
            sav = np.fromiter(map(lambda jk : costsav(jk[0],jk[1]), product(S,N_S)),dtype=float)
            # Obtain maximal saving
            jk = sav.argmax()
            if not sav[jk] > 0.0: break
            j, k = np.unravel_index( jk, shape=(len(S),len(N_S)) ) 
            # Open facility j_a = N_S[k] and close j_d = S[j]
            itr += 1
            j_d, j_a = S[j], N_S[k] 
            objv -= sav[jk]
            S[j], N_S[k] = j_a, j_d
            cmin1, cmin2 = np.partition( c[S], kth=1, axis=0 )[0:2,:]
            if not self.silent: print('{0:4d}   {1:15.2f}'.format(itr,objv) )
                    
        self._set_comptime()
        self.cost  = objv
        self.fcost = f[S].sum()
        self.tcost = self.cost - self.fcost
        self.facilities = S 
        self.assigned = list(map(lambda j : S[j], np.argmin(c[S], axis=0)))
        self.itr = itr 
        
        if not self.silent: print('-'*30)
    
        
    #---------------------------------------------
    
    def __LR_subgr( self ):
        """ 
        Lagrangian relaxation and heuristic for the UFLP. 
        """        
        c, f = self.c, self.f 
        n, m = c.shape 
        if not self.silent:
            print('-'*70)
            print('Iter   Lower bound   Upper bound   Best lower bound   Best upper bound')
            print('-'*70)
    
        self._set_starttime( )
        
        # Function to determine the supergradient 
        get_gradient = lambda dualv, S : 1- ( c[S] < dualv ).sum(axis=0)
        
        # Lower bounds and upper bounds on Lagrangian multipliers
        maxLam = c.max( axis = 0 )
        minLam = np.partition( c, kth=1, axis=0 )[1,:] #2nd smallest in each column

        # Initial feasible primal solution
        self.silent, shut_up = True, self.silent  
        self.__addProc( )
        self.silent = shut_up 
        primal_improve = False
    
        # Initialize Lagrangian multipliers
        dualv = np.fromiter( (map(lambda i : c[self.assigned[i],i], range(m))), dtype=float ) 

        # Initialize step size parameters
        alpha = self.__alpha
        maxIter = max( 30, self.__miter*(m+n) )
        H = max( self.__half, int( 0.03*maxIter ) )
     
        # Perform subgradient steps
        nfail, itr, lobnd = 0, 0, 0.0
        for itr in range(maxIter):    
            # Solve the Lagrangian suproblem
            rho = np.maximum(0, dualv - c).sum(axis=1) - f 
            S = list(filter( lambda j : rho[j] > 0.0, range(n)))
            if len(S)==0: S=[np.argmax(rho)]
            cur_bnd = dualv.sum() - rho[S].sum()
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
            U = (c[S].min(axis=0)).sum() + f[S].sum()
            if ( U < self.cost  ):
                self.cost = U
                self.facilities = S.copy()    
                primal_improve = True
            # Show progress
            if not self.silent:
                print('{0:4d}   {1:11.2f}   {2:11.2f}   {3:16.2f}   {4:16.2f}'.format(itr,cur_bnd,U,lobnd,self.cost) )
            # Stop if epsilon-optimal solution reached
            if self.cost/max(1.0,lobnd) - 1.0 < self.__epsi: break
            # Obtain the subgradient
            g = get_gradient( dualv, S ) 
            # Compute step size
            theta = alpha*( self.cost - cur_bnd )/np.inner(g,g)
            # Apply subgradient step
            dualv = np.minimum( np.maximum( dualv + theta*g, minLam ), maxLam )

        if not self.silent: print('-'*70)
        if primal_improve: 
            self.assigned = list(map(lambda j : S[j], np.argmin(c[S], axis=0)))
            
        self._set_comptime( )
        self.itr = itr 
        self.bound = lobnd 
    
    #---------------------------------------------
    
    def solve( self, method='LR' ):
        """
        Obtain a solution to the UFLP using method 'method'. 

        Parameters
        ----------
        method : str 
             Determines the method to be applied. 

             * 'LR'  : Lagrangian heuristic 
             * 'ADD' : Greedy Add heuristic 
             * 'DROP': Greedy Drop heuristic
             * 'AI'  : Add followed by Interchange
             * 'DI'  : Drop followed by Interchange 
        """ 
        meth = method.upper()
        if 'LR' in meth:
            self.__LR_subgr()
        elif 'ADD' in meth: 
            self.__addProc() 
        elif 'DROP' in meth:
            self.__dropProc()
        elif 'AI' in meth: 
            self.__addProc() 
            self.__interchg()
        elif 'DI' in meth:
            self.__dropProc()
            self.__interchg()

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
