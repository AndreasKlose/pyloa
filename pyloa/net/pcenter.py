"""
    Module net.pcenter of package pyloa:
    Methods for solving the vertex p-center problem.
"""
import numpy as np
from itertools import combinations
from pyloa.mip.model import Model
from pyloa.net.netprob import NetProblem

#------------------------------------------------------------------------------

class PCenter( NetProblem ):

    def __init__( self, fname=None, dstmat='matrix', orlib=False, d=None, 
                  dmax=None, m=0, n=0, w=None, p=None ):
        """
        Creates an instance of the class PCenter.
        
        Parameters
        ----------   
        see base class NetProblem in module pyloa.net_prob for a description
        of the parameters.
        """
        super().__init__(fname=fname, dstmat=dstmat, orlib=orlib, d=d, m=m, n=n, w=w, p=p)
                       
        self.radius = 0 
        """Largest weighted distance of a customer to a facility in a solution"""
                    
        self.__epsi = 1.0E-05 
        """Tolerance value"""
        
        self.__x_continuous = True 
        """True if allocation variables are treated as continuous variables"""
        
    #---------------------------------------------

    def __set_solution( self, S, objv=None ):
        """
        Register solution with facilities from S
        """
        self.facilities = S
        self.assigned = self.get_assigned( S )
        if objv is None: 
            w_dists = self.min_wdist( S )
            self.wdist = w_dists.sum( )
            self.radius = w_dists.max()
        else: 
            self.wdist = self.min_wdist(self.facilities).sum()
            self.radius = objv 
       
    #---------------------------------------------

    def __open_all( self ):
        """
        Returns solution to if all facility nodes are opened.
        """
        self.__set_solution(list(range(self._n)) )
 
    #---------------------------------------------
    
    def __one_center( self ):
        """
        Return the optimal 1-vertex center
        """
        if self._is_matrix:
            self.__set_solution([((np.vstack(self._w)*self._dmat).max(axis=0)).argmin()])
        else:
            j = min(range(self._n), key = lambda j : max(self._w[i]*self._d(i,j) for i in range(self._m) ) )
            self.__set_solution([j])      

#---------------------------------------------
     
    def __trad_pcenter( self, keep_model = False ):
        """
        Formulate vertex p-center in a traditional way as a MILP 
        and solve it by means of a MIP solver.
        """
        if not self.model is None: self.model.end()
        
        m, n, p, w, d = self._m, self._n, self._p, self._w, self._d 
        M = Model('P-center problem (traditional)')
        self.model = M 
        
        # Allocation/assignment variables
        x = M.addVarMatrix(m, n) if self.__x_continuous else M.addVarMatrix(m, n, vtype='B')
    
        # Location variables
        y = M.addVars( n, vtype='B' )
    
        # Variable for the radius
        r = M.addVar()
    
        # Assignment constraints    
        M.addConstraints( M.sum(x[i,j] for j in range(n))==1 for i in range(m) )
                           
        # No assignment to a non-center node
        M.addConstraints( x[i,j]-y[j] <= 0 for i,j in x.keys() )
    
        # Number of center nodes
        M.addConstraint( M.sum( y ) == p )
    
        # Constraints to model the radius of a solution
        M.addConstraints( r >= M.sum( x[i,j]*w[i]*d(i,j) for j in range(n)) for i in range(m) )
    
        # Objective function
        M.minimize( r )
                                 
        # Solve model
        self._set_starttime( )
        M.log_output = not self.silent 
        if M.optim( ):
            self.bound = M.ObjBound 
            self._set_comptime( mip_time=True )
            self.__set_solution( list(M.get_solution( y, keep_zeros=False, precision=0.1).keys()),\
                                 M.ObjVal )
        
        if keep_model:
            self.x = x 
            self.y = y
        else:      
            self.model.end()

    #---------------------------------------------
   
    def __get_wdist_list( self ):
        """
        Return sorted list of different weighted distance values
        """
        if self._is_matrix:
            return np.unique( np.vstack(self._w)*self._dmat )
        D = list(self._w[i]*self._d(j,i) for i,j in combinations(range(self._m),2))
        return np.unique( D )
     
#------------------------------------------------------------------------------

    def __elloumi_MIP( self, keep_model=False ):
        """
        Use Elloumi et al.'s formulation of the vertex p-center and solve it by
        means of a MIP solver.
        """
        if not self.model is None: self.model.end() 
       
        m, n, p, w, d = self._m, self._n, self._p, self._w, self._d 
        M = Model('P-center problem (Elloumi)')
        self.model = M
    
        # Obtain sorted list of weighted distances
        C = self.__get_wdist_list( )
   
        # 'Covering' variables z
        K = C.shape[0]
        z = M.addVars(K) if self.__x_continuous else M.addVars(K,vtype='B')
        z[0].lb = 1 
        z[0].ub = 1
    
        # Location variables
        y = M.addVars( n, vtype='B' )
    
        # Number of center nodes
        M.addConstraint( M.sum(y) == p )
    
        # Covering constraints
        M.addConstraints( z[k] + M.sum( y[j] for j in range(n) if w[i]*d(i,j) < C[k]) >= 1 \
                          for k in range(1,K) for i in range(m) )
                               
        # Objective function
        M.minimize( M.sum( (C[k]-C[k-1])*z[k] for k in range(1,K) ) )
                                 
        # Solve model
        self._set_starttime( )
        M.log_output = not self.silent 
        if M.optim(  ): 
            self.bound = M.ObjBound
            self._set_comptime( mip_time=True )
            self.__set_solution(list(M.get_solution( y, keep_zeros=False, precision=0.1).keys()))
        
        if keep_model:
            self.z = z 
            self.y = y
        else:      
            self.model.end() 
            
    #---------------------------------------------
       
    def __build_scp_model( self, r ):
        """
        Set up the MIP model of the linearly relaxed set 
        covering problem to be repeatedly solved in Elloumi
        et al.'s method. Parameter r is thereby the current
        trial radius.
        """
        m, n = self._m, self._n 
        M = Model("Set covering")
        y = M.addVars( self._n )
        M.minimize( M.sum(y) )
        constr = M.addConstraints(( M.sum( y[j] for j in range(n) if self._w[i]*self._d(i,j) <= r) >= 1 \
                                    for i in range(m)), return_constr = True )
        M.y = y
        M.constr = constr
        M.log_output = False

        return M
        
    #----------------------------------------------
    
    def __adjust_scp( self, r, scp ):
        """
        Adjust the covering constraints to the new radius r in the current
        set covering model scp.
        """
        for i, c in enumerate(scp.constr):
            coeffs = zip( scp.y.values(), (int(self._w[i]*self._d(i,j) <= r) for j in range(self._n) ) )
            scp.chgCoeffs( c, coeffs )
            
    #----------------------------------------------
    
    def __sortLP( self, scp ):
        """
        Obtain p-center solution from current solution, s, to the linear 
        set covering problem, scp, by opening the facilities showing 
        largest solution value of the corresponding location variable. 
        If this results in less than p open facilities, open additional 
        ones showing smallest LP reduced costs.
        """
        y = np.fromiter( scp.get_solution( scp.y ).values(), dtype=float )
        fac_lst = np.argsort(-y)
        j = self._p
        while y[fac_lst[j-1]] < 1.0E-5: j -= 1
        if j < self._p:
            red_cst = scp.reduced_costs( scp.y.values() )
            fac_lst[j:] = sorted(fac_lst[j:], key = lambda k : red_cst[k])
        S = fac_lst[:self._p]
        return (self.min_wdist(S)).max(), S

    #----------------------------------------------
    
    def __elloumi( self ):
        """
        Implements Elloumi et al.'s set covering based solution 
        method for the p-center problem. Phase 1 of the method 
        computes a lower bound on the minimal radius by solving 
        a sequence of linear set covering problems.Starting with 
        this lower bound, the second phase then obtains the
        minimal radius and corresponding vertex p-center using a 
        binary search that in each iteration solves an (integer) 
        set covering problem.
        """    
        # Sorted list of weighted distances
        C = self.__get_wdist_list( )
    
        # First phase: Find lower bound on the minimal radius
        l = -1       # strict lower bound
        u = len(C)-1 # upper bound on (non-integer) minimal radius
        U = C[-1]    # upper bound on integer minimal radius
        k = (l+u)//2 # C[k] curent trial radius
    
        scp = self.__build_scp_model( C[k] )
 
        if not self.silent:
            print('-'*56)
            print("Elloumi's method for solving the vertex p-center problem")
            print('-'*56)
            print("Phase   Iter   lower bound   upper bound   best radius")
            print('-'*56)
    
        self._set_starttime( )
        itr = 0 
        while True:
            itr += 1
            # Solve the linear set covering problem
            if not scp.optim( ): break 
            num_facs = scp.ObjVal  
            if num_facs > self._p:
                l = k
            else:
                u = k
            # Apply simple heuristic to get a p-center solution
            objv, facLst = self.__sortLP( scp )
            if objv < U:
                U = objv
                self.facilities = facLst
            if not self.silent: 
                L = -1 if l < 0 else C[l]
                print('{0:5d}   {1:4d}   {2:11.0f}   {3:11.0f}   {4:11.0f}'.format(1,itr,L,C[u],U) )       
            k = (l+u)//2
            if l+1 == u: break
            # Adjust the covering constraints to the new radius C[k]
            self.__adjust_scp( C[k], scp )       
        
        # Second phase: Binary search for minimal radius r* over (C[l], U)
        l = k
        u = np.searchsorted(C,U) if U < C[-1] else len(C)-1    
        itr = 0
             
        # Upper cutoff value and lower stop value
        scp.upper_cutoff = self._p + self.__epsi
        scp.lowerObjStop = self._p 
        
        # Make all variables binary
        for y in scp.y.values(): scp.set_vartype(y, 'B')
        while (l+1) < u:
            itr += 1
            k = (l+u)//2
            self.__adjust_scp( C[k], scp )
            if not scp.optim():
                # Due to the upper cut-off no solution of objective larger p possible
                l = k
            else:
                u = k
                U = C[u]
                self.facilities = list(scp.get_solution(scp.y, keep_zeros=False, precision=0.1).keys())
            if not self.silent:
                print('{0:5d}   {1:4d}   {2:11.0f}   {3:11.0f}   {4:11.0f}'.format(2,itr,C[l],C[u],U) )
      
        scp.end()      
        if not self.silent: print('-'*56)
        self.bound = U
        self.__set_solution(self.facilities, U)
        self._set_comptime()
        
    #----------------------------------------------
    
    def pcp_solve(self, method='Elloumi', keep_model=False ):
        """
        Computes a solution to the vertex p-center problem.
        
        Parameters
        ----------
        method : str 
            Determines the solution method to be employed:
            
            method = 'Elloumi'
                Use the set-covering based solution algorithm
                suggested by Elloumi, Labbe and Pochet.
            
            method = 'MIP-Traditional'
                Use the traditional MIP formulation and solve it  
                using a MIP solver. Allocation variables are declared 
                as continuous.
                
            method = 'IP-Traditional'
                Same as MIP-Traditional but allocation variables
                are declared as binary.
                
            method = 'MIP-Elloumi'
                Solves Elloumi et al.'s formulations where variables
                z are declared as continuous.
                
            method = 'IP-Elloumi'
                Solves Elloumi et al.'s formulations where variables
                z are declared as binary.

        keep_model : bool
            If True, the MIP model is kept for later purposes,
            otherwise destroyed after the computations. This option
            only applies if a MIP solver is used as solver.    
        """
        if self._p >= self._n: 
            self.__open_all( )
        elif self._p==1: 
            self.__one_center( )
        else: 
            meth = method.lower()
            if meth == 'elloumi':
                self.__elloumi()
            else:
                self.__x_continuous = 'mip' in meth      
                if 'trad' in meth: 
                    self.__trad_pcenter(keep_model=keep_model)
                else: 
                    self.__elloumi_MIP(keep_model=keep_model)    
    
    #----------------------------------------------
