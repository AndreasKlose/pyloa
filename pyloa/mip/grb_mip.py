"""
    Module mip.GRBmodel of package pyloa:
    Defines wrapper class for the gurobipy Model class.
"""
from gurobipy import Model, Column, GRB, quicksum as qsum
            
#--------------------------------------------------------------------------

class GRBmodel:
    """
    Wraps some methods and attributes of the gurobipy model class for
    the purposes of synchronizing function names with docplex.      
    """
    def __init__(self, name=None  ):
        """
        Parameters
        ----------
        name : str, optional
            Name of the model, default is None
        """             
        self.__model = Model() if name is None else Model(name)
        """Instance of the gurobipy Model class"""
        
        self.__callbck = None 
        """Instance of a callback class"""
        
        self.__wheres = None 
        """List of wheres when to to use a callback function"""
        
    #---------------------------------------------
    
    def set_callback( self, callbck, contxtmsk=None, wheres=None ):
        """
        Set callback class for the purposes ('wheres') defined
        by wheres.
        
        Parameters
        ----------
        callbck : class 
            The class to be called for the callback
        contxtmask : int 
            Cplex's indicator for the context in which the
            callback might be called.
        wheres : list of int
            GuRoBi's list of 'where' codes to be enabled in
            the callback.
        """
        self.__callbck = callbck 
        self.__wheres = wheres
            
    #---------------------------------------------
    
    def in_relaxation(self, where ):
        """
        Return True if GuRoBi solver calls callback
        after solving the relaxation.
        """
        return where==GRB.Callback.MIPNODE and \
               self.__model.cbGet(GRB.Callback.MIPNODE_STATUS)==GRB.OPTIMAL
     
    #---------------------------------------------
    
    def cbGetNodeRel( self, dvars ):
        """
        Return the relaxation's solution for the decision variables 
        dvars at the current branch-and-cut node.
        
        Parameters
        ----------
        dvars : sequence of decision variables
        
        Returns
        -------
        List of solution values to the specified variables
        """
        return self.__model.cbGetNodeRel( dvars )
     
    #---------------------------------------------
    
    def cbCut(self, cut ):
        """
        Add a global user cut from a callback to the solver.
        
        Parameters
        ----------
        cut : GuRoBi linear constraint 
           The object representing the cut as a
           linear constraint. In the form
           lhs R rhs, where R is <=, =, or >=
        """
        self.__model.cbCut( cut )
  
    #---------------------------------------------
     
    def addConstraint(self, constr, return_constr=False ):
        """
        Adds the single constraint constr to the model.
        
        Parameters
        ----------
        constr : GuRoBi temporary constraint object
            The constraint to be included to the model
        
        Returns
        -------
        cnstr : gurobipy.tempConstr
            The temporary form of the constraint included to the model.
        return_constr : bool 
            If True, the constraint object instance
            is returned.
        """             
        c = self.__model.addConstr(constr)
        if return_constr: return c
              
    #---------------------------------------------
    
    def addConstraints(self, constrs, return_constr=False ):
        """
        Adds several constraints to the model.
        
        Parameters
        ----------
        constrs : iterable of gurobipy.Constr or gurobipy.tempConstr    
            The iterable or generator expression of constraints to be included.        
        return_constr : Bool
            If True, the list of the added constraints is returned.
            
        Returns
        -------
        None or the list of the added constraints. Note that this
        are temporary constraints in GuRoBi until the model is updated.
        """   
        c = self.__model.addConstrs( constrs )
        if return_constr: return list( c.values() )
        
    #---------------------------------------------
    
    def addQuadConstrs(self, constrs, return_constr=False):
        """
        Add a bunch of quadratic constraints to the model.
        Is actually the same as addConstraints as GuRoBi
        does not care if the constraints are linear or
        quadratic.
        
        Parameters
        ----------
        constrs: iterable of docplex.mp.constr.QuadraticConstraint    
            The quadratic constraint expressions.    
        return_constr : Bool
            If True, the added constraints are returned.
        """
        c = self.__model.addConstrs( constrs )
        if return_constr : return c
      
    #---------------------------------------------
    
    def minimize (self, expr ):
        """
        Sets the objective function of the underlying GuRoBi model 
        to be the minimization of the expression expr. This is
        an alias to gurobipy.model.SetObjetive(expr, GRB.MINIMIZE)
                
        Parameters
        ----------
        expr : gurobi LinExpr
            single objective function to be minimized.
        """
        self.__model.setObjective(expr,GRB.MINIMIZE)
        
    #---------------------------------------------
    
    def maximize (self, expr ):
        """
        Sets the objective function of the underlying GuRoBi model 
        to be the maximization of the expression expr. This is
        an alias to gurobipy.model.SetObjetive(expr, GRB.MAXIMIZE)
                
        Parameters
        ----------
        expr : gurobi LinExpr
            single objective function to be minimized.
        """
        self.__model.setObjective(expr,GRB.MAXIMIZE)
             
    #------------------s---------------------------
    
    def __getObjVal(self):
        """
        Return the objective value of the solution
        to the underlying GuRoBi MIP model (and
        None if there is no solution).
        """
        try:
            obj = self.__model.ObjVal 
        except:
            return None
        return obj 
       
    #---------------------------------------------
    
    def addVar(self, lb=0.0, ub=float('inf'), obj=0.0, vtype=None,\
               name='', constr=None, coeffs=None ):
        """
        Adds a single variable to the model.
        
        Parameters
        ----------
        lb : float
            Lower bound for the variable. Default: 0.0
        ub : float
            Uppper bound on the variable. Default: None
        obj : float
            The variables objective coefficient
        vtype : None or str
            If None, a continuous variable is created. 
            Otherwise, if vtype='B' a binary and if
            vtype='I' an integer variable.
        name : str 
            Name to be given to the variable. Default is None.
        constr : None or a sequence of constraints
            Sequence of constraint to which the variable should 
            be added.
        coeffs : None or a single float/int or an iterable of float/int.
            If coeffs is a single float or int, the variable
            is added to all constraints from the above sequence
            of constraints with the same coefficient 'coeffs'.
            If coeffs is an iterable, the variable is included
            with the specified coefficient to the given sequence
            of constraints. Make sure that both sequences are
            of the same length.
            
        Returns
        -------
        An instance of class Var (gurobipy)
        """
        col= None  
        if not (constr is None or coeffs is None): 
            col = Column(list(coeffs),list(constr)) if hasattr(coeffs,'__iter__') \
                  else Column( [coeffs]*len(constr),list(constr) )
            
        if vtype is None:
            return self.__model.addVar( lb=lb, ub=ub, obj=obj, vtype=GRB.CONTINUOUS,\
                                       name=name, column=col )
        if vtype.upper()=='B': 
            return self.__model.addVar(vtype=GRB.BINARY, obj=obj, name=name, column=col )
        return self.__model.addVar(lb=lb, ub=ub, obj=obj, vtype=GRB.INTEGER, name=name,\
                                   column=col)
     
    #---------------------------------------------
    
    def addVars(self, indices, lb=0.0, ub=float('inf'), vtype=None, name=None ):
        """
        Adds a dictionary of variables with keys
        indices to the MIP model.
        
        Parameters
        ----------
        indices : iterable 
            Sequence of indices to be used as 
            keys of the dictionary.
        lb : int or float or iterable of int or float
            Lower bounds to be applied for the variables.
            If None, lower bounds are zero.
        ub : int or float or iterable of int or float
            Upper bounds to be applied for the variables.
            If None, no upper bounds are imposed.
        vtype : None or str
            Type of the variables. If None, it's
            continuous variables, if 'B' binary
            variables, if 'I' integer variables.
        name : None or str 
            If not None, variables are named as name[index] 
            
        Returns
        -------
        v : dict
            The dictionary of MIP model variables.
        """
        nme = '' if name is None else name
        if vtype is None: return self.__model.addVars( indices, lb=lb, ub=ub, name=nme )
        if vtype.upper()=='B': 
            return self.__model.addVars( indices, vtype=GRB.BINARY, name=nme )
        return self.__model.addVars( indices, lb=lb, ub=ub, vtype=GRB.INTEGER, name=nme )    
    
    #---------------------------------------------
    
    def addVarMatrix(self, rows, cols, lb=0.0, ub=float('inf'), vtype=None, name=None ):
        """
        Create and return a rows x cols matrix of variables
        of type vtype.
    
        Parameters
        ----------
        rows, cols : int 
            Number of rows and columns of the matrix. 
        lb : int or float or iterable of int or float
            Lower bounds to be applied for the variables.
            If None, lower bounds are zero.
        ub : int or float or iterable of int or float
            Upper bounds to be applied for the variables.
            If None, no upper bounds are imposed.
        vtype : None or str
            Type of the variables. If None, it's
            continuous variables, if 'B' binary
            variables, if 'I' integer variables.
        name : None or str 
            If not None, variables are named as name[index] 
            
        Returns
        -------
        v : dict
            The dictionary of MIP model variables.
        """
        nme = '' if name is None else name
        if vtype is None: 
            return self.__model.addVars( rows, cols, lb=lb, ub=ub, name=nme ) 
        if vtype.upper()=='B': 
            return self.__model.addVars( rows, cols, vtype=GRB.BINARY, name=nme )
        return self.__model.addVars( rows, cols, lb=lb, ub=ub, vtype=GRB.INTEGER, name=nme )
    
    #---------------------------------------------
    
    def addVarCube(self, indx, lb=0.0, ub=float('inf'), vtype=None, name=None):
        """
        Creates and return a dictionary of variables indexed by the
        the three indices.
    
        Parameters
        ----------
        indx : tuple of three int 
            tuple of the three indices 
        lb : int or float or iterable of int or float
            Lower bounds to be applied for the variables.
            If None, lower bounds are zero.
        ub : int or float or iterable of int or float
            Upper bounds to be applied for the variables.
            If None, no upper bounds are imposed.
        vtype : None or str
            Type of the variables. If None, it's
            continuous variables, if 'B' binary
            variables, if 'I' integer variables.
        name : None or str 
            If not None, variables named as name[index] 
            
        Returns
        -------
        v : dict
            The dictionary of MIP model variables.
        """
        if vtype is None: 
            return self.__model.addVars(indx[0], indx[1], indx[2], lb=lb, ub=ub, name=name) 
        if vtype.upper()=='B': 
            return self.__model.addVars( indx[0], indx[1], indx[2], name=name )
        return self.__model.addVars( indx[0], indx[1], indx[2], lb=lb, ub=ub, name=name )   
    
    #---------------------------------------------
    
    def set_vartype(self, dvar, vtype ):
        """
        Set the type of the decision variable var
        to the type vtype.
        
        Parameters
        ----------
        dvar : gurobipy.Var object
            The variable whose type should be set.
        vtype : str 
            The type to which the variables should be set:
            'B' for binary, 'I' for integer, 'C' for 
            continuous.
        """
        dvar.Vtype = vtype 
     
    #---------------------------------------------
    
    
    def sum(self, exprs ):
        """
        Returns the sum of a list or iterable of 
        expressions using gurobipy's quicksum
        """
        return qsum(exprs)
    
    #---------------------------------------------
    
    def optim( self ):
        """
        Calls the solver (gurobipy.model.optimize) for
        solving the current MIP model.
                               
        Returns
        -------
        success : bool 
            True, if a feasible or optimal solution was found.
        """
        if self.__wheres is None:
            self.__model.optimize( self.__callbck )
        else:
            # Requires GuRoBi 13
            self.__model.optimize( self.__callbck, wheres=self.__wheres )
        return self.__model.SolCount > 0
        
    #---------------------------------------------
    
    def get_solution( self, X, keep_zeros=True, precision=1.0E-06 ):
        """
        Returns the solution values of variables X.
        
        Parameters
        ----------
        X : dict
            Dictionary of docplex.mp.dvar
        keep_zeros : bool
            If True, the default, all solution 
            values to variables X are returned.
            Otherwise, only non-zero values are
            returned.
        precision : float (positive)
            Figures smaller in magnitude than precision are seen as
            zeros.
        """
        try:
            sol = self.__model.getAttr("X",X)
            if keep_zeros: return sol 
            return dict(filter(lambda itm : abs(itm[1]) >= precision, sol.items()))
        except:
            return None
        
    #---------------------------------------------
    
    def get_dual_values(self, constr ):
        """
        Returns the list/sequence of dual variables to
        the linear constraints in the list/sequence
        of linear constraints constr. This is an alias
        for docplex.mp.model.dual_values()
        """
        return list( c.pi for c in constr )
        
    #---------------------------------------------
    
    def getVars(self):
        """
        Alias for gurobipy.model.getVars, which returns
        a list of the model's variables.
        """
        return self.__model.getVars()
    
    #---------------------------------------------
    
    def getConstraints(self):
        """
        Alias for gurobipy.model.getConstr, which returns
        a list all linear constraints in the model.
        """
        return self.__model.getConstrs()
    
    #---------------------------------------------

    def getObjective(self):
        """
        Alias for gurobioy.model.getObjective, which
        returns the objective expression.
        """
        return self.__model.getObjective()
            
    #---------------------------------------------
    
    def reduced_costs(self, dvars ):
        """
        Returns the reduced cost of the decision variables
        dvars.
        
        Parameters
        ----------
        dvars: iterable/sequence of variables
            sequence of decision variables
        
        Returns
        -------
        rc : list of float
            The list of the variables' reduced cost
    
        """
        return list( map( lambda x : x.RC, dvars ) )
    
    #---------------------------------------------
    
    def chgCoeff(self, constr, dvar, coef=0.0 ):
        """
        Change the coefficient of variable dvar in
        the linear constraint constr to value coef.
        
        Parameters
        ----------
        constr : gurobipy Constr
            The constraint to be adjusted
        dvar : gurobipy Var
            The decision variable where to change
            the coefficient.
        coef : int or float
            The value of the (new) coefficient.
        """
        self.__model.chgCoeff(constr, dvar, coef)
   
    #---------------------------------------------
    
    def chgCoeffs(self, constr, coeffs ):
        """
        Changes coefficients of a number of variables
        in the linear constraint constr.
        
        Parameters
        ----------
        coeffs : iterable/sequence of pairs
             iterable/sequence of variable-coefficient pairs
        constr : gurobipy.Constr
            The constraint to be adjusted
        """
        for d,c in coeffs: self.__model.chgCoeff(constr, d, c)
         
    #---------------------------------------------
      
    def export_model ( self, lp_file ):
        """
        Exports the model to a model file of type "lp". 
        
        Parameters
        ----------
        lp_file : str    
            Name of the file where to write the model.
        """
        self.__model.write( lp_file )
        
    #---------------------------------------------
    
    def update(self):
        """
        gurobipy.model.update
        """
        self.__model.update() 
       
    #---------------------------------------------
    
    def end ( self ):
        """
        Clean up the GuRoBi IP model by calling its "close" method.
        """
        if not self.__model is None: self.__model.close()
        self.__model = None
    
    #---------------------------------------------
    
    def terminate(self):
        """
        Terminate GuRoBi's computation on the model from a callback
        """
        self.__model.terminate()
        
    #---------------------------------------------
    
    @property 
    def mipModel(self):
        """
        Return the instance of the gurobipy Model class.
        """
        return self.__model 
    
    @property
    def ObjVal(self):
        """Objective value of a solution to the model"""
        return self.__getObjVal()
    
    @property
    def ObjBound(self):
        """
        Best bound on (lower for minimization, upper for
        maximization) on the optimal objective value
        """
        return self.__model.ObjBound 
    
    @property
    def runtime(self):
        """Return computation time used to solve the model"""
        return self.__model.runtime
    
    @property 
    def dettime(self):
        """Return the deterministic time. For GuRoBi, this
        is the model attribute Work, a figure that roughly
        corresponds to a second on a single thread"""
        return self.__model.Work
    
    @property
    def status(self):
        """Return the model status, cf. gurobipy.Model.status"""
        return self.__model.status    
    
    @property 
    def nodeCount(self):
        """Return the NodeCount, i.e. the number of branch-and-cut 
        nodes explored in GuRoBi's most recent optimization"""
        return self.__model.NodeCount 
    
    @property 
    def infinity(self):
        return GRB.INFINITY
            
    @property 
    def MIPGapAbs(self):
        """Return parameter MIPGapAbs"""
        return self.__model.params.MIPGapAbs
    
    @MIPGapAbs.setter 
    def MIPGapAbs(self, value ):
        """Set parameter MIPGapAbs"""
        if value >= 0.0: self.__model.params.MIPGapAbs = value
    
    @property 
    def MIPGap(self):
        """Return parameter MIPGap"""
        return self.__model.params.MIPGap
    
    @MIPGap.setter 
    def MIPGap(self,value):
        """Set parameter MIPGap"""
        if value >= 0.0: self.__model.params.MIPGap = value 
    
    @property 
    def IntFeasTol(self):
        """Return parameter IntFeasTol"""
        return self.__model.params.IntFeasTol
    
    @IntFeasTol.setter 
    def IntFeasTol(self, value):
        """Set parameter IntFeasTol"""
        if 0 <= value < 1: self.__model.params.IntFeasTol=value
    
    @property
    def FeasibilityTol(self):
        """Return parameter FeasibilityTol"""
        return self.__model.params.FeasibilityTol
    
    @FeasibilityTol.setter 
    def FeasibilityTol(self,value):
        """Set parameter FeasibilityTol"""
        if 0 <= value < 1: self.__model.params.FeasibilityTol = value 
    
    @property 
    def MIQCPMethod(self):
        """Returns value of parameter mip.strategy.miqcpstrat"""
        return self.__model.params.MIQCPMethod
    
    @MIQCPMethod.setter
    def MIQCPMethod(self, strategy ):
        """Sets parameter mip.strategy.miqcpstrat to value strategy-1"""
        self.__model.params.MIQCPMethod = strategy - 1
        
    @property 
    def timelimit(self):
        """Time limit for the MIP solver"""
        return self.__model.params.timelimit  
    
    @timelimit.setter 
    def timelimit(self, value):
        """Set the time limit for the MIP solver"""
        self.__model.params.timelimit = value if value > 0 else float('inf')   

    @property 
    def nodelimit(self):
        """Node limit for the MIP solver"""
        return self.__model.params.nodelimit  
    
    @nodelimit.setter 
    def nodelimit(self, value):
        """Set the time limit for the MIP solver""" 
        self.__model.params.nodelimit = value if value >= 0 else float('inf')   
        
    @property 
    def upper_cutoff( self ):
        """Return (upper) cutoff value."""
        return self.__model.params.cutoff
    
    @upper_cutoff.setter 
    def upper_cutoff( self, value ):
        """Set an upper cutoff value."""
        self.__model.params.cutoff = value
        
    @property 
    def lower_cutoff( self ):
        """Return lower cutoff value."""
        return self.__model.params.cutoff
    
    @lower_cutoff.setter 
    def lower_cutoff( self, value ):
        """Set a lower cutoff value."""
        self.__model.params.cutoff = value
    
    @property 
    def lowerObjStop(self):
        """
        Returns GuRoBi parameter BestObjStop (for minimization) 
        """
        return self.__model.params.BestObjStop
    
    @lowerObjStop.setter 
    def lowerObjStop(self, value ):
        """
        Set GuRoBi parameter BestObjStop to the given value.
        """
        self.__model.params.BestObjStop = value 
   
    @property
    def log_output(self):
        """Return True if value of gurobiy.model.OutputFlag=1"""
        return self.__model.params.OutputFlag > 0

    @log_output.setter
    def log_output(self, on_off):
        """If True, gurobipy.model.OutputFlag is set to 1"""
        self.__model.params.OutputFlag = int(on_off)
 
    @property 
    def mipStrategy(self):
        """Dynamic search switch, does not exist for GuRoBi"""
        return 0
    
    @mipStrategy.setter 
    def mipStrategy(self, value ):
        """Dynamic searc switch, does not exist for GuRoBi"""
        pass 
        
    @property 
    def mipEmphasis(self):
        """Return value of GuRoBi parameter MIPFocus"""
        return self.__model.parameters.MIPFocus
    
    @mipEmphasis.setter 
    def mipEmphasis(self, value ):
        """Set GuRoBi parameter MipFocus to value. Mimic thereby the
        settings for Cplex: 0 = Balanced (default), 1 = Emphasise 
        feasibility, 2 = Emphasis optimality, 3 = Emphasis the bound,
        4 = HIDDENFEAS does not exist for GuRoBi."""
        if value in (0,1,2,3): self.__model.parameters.mipFocus=value
    
    @property 
    def lbHeur(self):
        """Dummy. GuRoBi has no local branching heuristic"""
        return 0
    
    @lbHeur.setter 
    def lbHeur(self, value ):
        """Dummy. GuRoBi has no local branching heuristic"""
        pass 
        
    @property 
    def cuts(self):
        """Return value of GuRoBi parameter cuts"""
        return self.__model.params.Cuts 
               
    @cuts.setter 
    def cuts( self, value ):
        """
        Set GuRoBi parameter Cuts to value, where -1 means automatic
        cut generation, 0 means cuts are switched off, 1 for
        moderate generation of certain cuts, 2 for agressive
        cut generation and 3 for very aggressive cut generation.
        """
        if value in (-1,0,1,2,3): self.__model.params.Cuts = value 
    
    @property
    def doBenders( self ):
        """Dummy. GuRoBi has no built-in Benders"""
        return -1
    
    @doBenders.setter 
    def doBenders(self, value ):
        """Dummy. GuRoBi has no built-in Benders"""
        pass 
       
    @property 
    def id_relaxation(self):
        """
        Return GRB.Callback.MIPNODE
        """
        return GRB.Callback.MIPNODE

    @property 
    def cb_MIPSOL(self):
        """Return the constant GRB.Callback.MIPSOL"""
        return GRB.Callback.MIPSOL
    
    @property
    def candidate_objective(self):
        """Return objective value of a candidate solution"""
        return self.__model.cbGet(GRB.Callback.MIPSOL_OBJ)
    
    @property
    def incument_objective(self):   
        return self.__model.cbGet(GRB.Callback.MIPSOL_OBJBST)
