"""
    Module mip.cpx_mip of package pyloa:
    Defines wrapper class for the docplex.mp.model Model class.
"""
import cplex
from docplex.mp.model import Model
from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin
             
#--------------------------------------------------------------------------

class CPXmodel:
    """
    Wraps some methods and attributes of the docplex.mp.model class for
    the purposes of synchronizing function names with GuRoBi. 
    """
    def __init__( self, name ):
        """
        Parameters
        ----------
        name : str, optional
            Name of the model, default is None
        """            
        self.__model = Model() if name is None else Model(name)
        """Instance of the docplex.mp.model class"""
        
        self.__s = None 
        """Docplex.mp.solution object"""
        
        self.__log_output = False 
        """If True, Cplex's optimizer will send log output to stdout"""
        
        self.__callbck = None 
        """Instance of a callback class"""
        
        self.__cuts = -1 
        """Global cuts parameter as for GuRoBi. Value of -1 means
        automatic cut generation"""
        
        self.__contxtmsk = None 
        """Context mask used by a callback function"""
        
        self.__context = None 
        """Instance of class cplex.callbacks.Context"""
                      
    #---------------------------------------------
    
    def set_callback( self, callbck, contxtmsk=None, wheres=None ):
        """
        Set callback class for the purposes defined
        by contxtmsk. 
        
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
        self.__contxtmsk = contxtmsk
        if not (callbck is None or contxtmsk is None):
            self.__model.cplex.set_callback( callbck, contxtmsk ) 
            
    #---------------------------------------------
    
    def in_relaxation(self, context ):
        """
        Return True if Cplex solver calls callback
        after solving the relaxation.
        """
        return context.in_relaxation()
        
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
        return self.__context.get_relaxation_point( dvars )
    
    #---------------------------------------------
    
    def cbCut(self, cut ):
        """
        Add a global user cut from a callback to the solver.
        
        Parameters
        ----------
        cut : docplex.mp.constr.LinearConstraint 
           The object representing the cut as a
           linear constraint.
        """
        lhs, sense, rhs = ModelCallbackMixin.linear_ct_to_cplex( cut )
        purge = cplex.callbacks.UserCutCallback.use_cut.purge 
        self.__context.add_user_cut( cut=lhs, sense=sense, rhs=rhs, \
                                     cutmanagement=purge, local=False )
    
    #---------------------------------------------
    
    def addConstraint( self, constr, return_constr=False ):
        """
        Adds the single constraint constr to the model.
        
        Parameters
        ----------
        constr : docplex.mp.constr.LinearConstraint  
            The constraint to be included 
        return_constr : bool 
            If True, the constraint object instance
            is returned.
        """             
        c = self.__model.add_constraint(constr)
        if return_constr: return c
              
    #---------------------------------------------
    
    def addConstraints(self, constrs, return_constr = False ):
        """
        Adds several constraints to the IP model.
        
        Parameters
        ----------
        constrs : iterable of docplex.mp.constr.LinearConstraint    
            The list of constraints to be included        
        return_constr : Bool
            If True, the list of added constraints is returned.
            
        Returns
        -------
        None or the list of the added constraints.
        """   
        c = self.__model.add_constraints( constrs )
        if return_constr: return c 
                 
    #---------------------------------------------
    
    def addQuadConstrs(self, constrs, return_constr=False):
        """
        Add a bunch of quadratic constraints to the model
        
        Parameters
        ----------
        constrs: iterable of docplex.mp.constr.QuadraticConstraint    
            The quadratic constraint expressions.    
        return_constr : Bool
            If True, the added constraints are returned.
        """
        c = self.__model.add_quadratic_constraints( constrs )
        if return_constr : return c
    
    #---------------------------------------------
    
    def minimize (self, expr ):
        """
        Sets the objective function of the underlying
        Cplex model to be the minimization of the 
        linear expression expr. This is just an
        alias to docplex.mp.model.minimize.
                
        Parameters
        ----------
        expr : docplex.mp.linear.LinearExpr
            single objective function to be minimized.
        """
        self.__model.minimize(expr)
    
    #---------------------------------------------
    
    def maximize (self, expr ):
        """
        Sets the objective function of the underlying
        Cplex model to be the maximization of the 
        linear expression expr. This is just an
        alias to docplex.mp.model.minimize.
                
        Parameters
        ----------
        expr : docplex.mp.linear.LinearExpr
            single objective function to be minimized.
        """
        self.__model.maximize(expr)
        
    #------------------s---------------------------
    
    def __getObjVal(self):
        """
        Return the objective value of the solution
        to the underlying docplex.mp MIP model (and
        None if there is no solution).
        """
        if self.__s is None: return None 
        return self.__model.objective_value
        
    #---------------------------------------------
    
    def addVar(self, lb=0.0, ub=None, obj=0.0, vtype=None, name=None,\
               constr=None, coeffs=None ):
        """
        Adds a single variable to the model.
        
        Parameters
        ----------
        lb : float
            Lower bound for the variable. Default: 0.0
        ub : float or None
            Uppper bound on the variable.
        obj : float
            The variables objective coefficient
        vtype : None or str
            If None, a continuous variable is created. 
            Otherwise, if vtype='B' a binary and if
            vtype='I' an integer variable.
        name : None or str 
            If not None, the variable name 
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
        A docplex.mp.dvar
        """ 
        if vtype is None:
            dvar = self.__model.continuous_var(lb=lb, ub=ub, name=name )
        elif vtype.upper()=='B': 
            dvar = self.__model.binary_var(name=name)
        else:
            dvar = self.__model.integer_var(lb=lb, ub=ub, name=name)
        
        if obj != 0.0: 
            objf = self.__model.get_objective_expr()
            objf += obj*dvar
        
        if constr is None or coeffs is None: return dvar
        
        if hasattr(coeffs, '__iter__'):
            for cnstr, coef in zip(constr,coeffs): cnstr.lhs += coef*dvar
        else: 
            for c in constr: c.lhs += coeffs*dvar
            
        return dvar 
            
    #---------------------------------------------
    
    def addVars(self, indices, lb=0.0, ub=None, vtype=None, name=None):
        """
        Adds a dictionary of variables with keys indices to 
        the MIP model.
        
        Parameters
        ----------
        indices : iterable 
            Sequence of indices to be used as 
            keys of the dictionary.
        lb : iterable of int or float
            Lower bounds to be applied for the variables.
            If None, lower bounds are zero.
        ub : iterable of int or float
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
        if vtype is None: 
            return self.__model.continuous_var_dict(indices, lb, ub, name=name) 
        if vtype.upper()=='B': return self.__model.binary_var_dict( indices, name=name )
        return self.__model.integer_var_dict( indices, lb, ub, name=name )    
    
    #---------------------------------------------
    
    def addVarMatrix(self, rows, cols, lb=0.0, ub=None, vtype=None, name=None):
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
            If not None, variables named as name[index] 
            
        Returns
        -------
        v : dict
            The dictionary of MIP model variables.
        """
        if vtype is None: 
            return self.__model.continuous_var_matrix(rows,cols, lb=lb, ub=ub, name=name) 
        if vtype.upper()=='B': return self.__model.binary_var_matrix( rows,cols, name=name )
        return self.__model.integer_var_matrix( rows, cols, lb=lb, ub=ub, name=name )    
    
    #---------------------------------------------
    
    def addVarCube(self, indx, lb=0.0, ub=None, vtype=None, name=None):
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
            return self.__model.continuous_var_cube(indx[0], indx[1], indx[2],\
                                                  lb=lb, ub=ub, name=name) 
        if vtype.upper()=='B': 
            return self.__model.binary_var_cube( indx[0], indx[1], indx[2], name=name )
        return self.__model.integer_var_cube( indx[0], indx[1], indx[2], lb=lb, ub=ub, name=name )    
    
    #---------------------------------------------
    
    def set_vartype(self, dvar, vtype ):
        """
        Set the type of the decision variable var
        to the type vtype.
        
        Parameters
        ----------
        dvar : docplex.mp.dvar object
            The variable whose type should be set.
        vtype : str 
            The type to which the variables should be set:
            'B' for binary, 'I' for integer, 'C' for 
            continuous.
        """
        vtype = vtype.upper()
        if vtype == 'B':
            Vtype = self.__model.binary_vartype
        else:
            Vtype = self.__model.integer_vartype if vtype=='I' else \
                    self.__model.continuous_vartype
        dvar.set_vartype(Vtype) 
    
    #---------------------------------------------
        
    def sum(self, exprs ):
        """
        Returns the sum of a list or iterable of 
        expressions using docplex.mp.model.sum
        """
        return self.__model.sum(exprs)
    
    #---------------------------------------------
    
    def mip_tol(self, absgap=-1.0, relgap=-1.0, int_tol=-1.0, feas_tol=-1.0 ):
        """
        Set the MIP solver's optimality absolute 
        and relative optimality tolerances to
        the values absgap and relgap, resp.
        
        Parameters
        ----------
        absgap, relgap, int_tol, feas_tol : float
            Non-negative floating point numbers. If negative, the 
            tolerance is not set. absgap and relgap are the
            absolute and relative optimality tolerance, int_tol
            is the tolerance for integrality, and feas_tol the
            feasibility tolerance for the simplex method.
        """
        if not absgap < 0.0:
            self.__model.parameters.mip.tolerances.absmipgap.set(absgap)
        if not relgap < 0.0:
            self.__model.parameters.mip.tolerances.mipgap.set(relgap)
        if not int_tol < 0.0:
            self.__model.parameters.mip.tolerances.integrality(int_tol)
        if not feas_tol < 0.0:
            self.__model.parameters.simplex.tolerances.feasibility(feas_tol)
    
    #---------------------------------------------
    
    def optim( self ):
        """
        Calls the Cplex solver (docplex.mp.model.solve) for
        solving the current MIP model.
                  
        Returns
        -------
        success : bool 
            True, if a feasible or optimal solution was
            found. 
        """
        self.__s = self.__model.solve(log_output=self.__log_output)
        return not self.__s is None
        
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
        if not self.__s is None:
            return self.__s.get_value_dict( X, keep_zeros=keep_zeros, precision=precision)
        
    #---------------------------------------------
    
    def get_dual_values(self, constr ):
        """
        Returns the list/sequence of dual variables to
        the linear constraints in the list/sequence
        of linear constraints constr. This is an alias
        for docplex.mp.model.dual_values()
        """
        return self.__model.dual_values( constr )
    
    #---------------------------------------------
        
    def getVars(self):
        """
        Returns a list of the model's variables.
        """
        return list( self.__model.iter_variables() )
    
    #---------------------------------------------
    
    def getConstraints(self):
        """
        Return list of all linear constraints in the model.
        """
        return list(self.__model.iter_linear_constraints())
    
    #---------------------------------------------
    
    def getObjective(self):
        """
        Alias for docplex.mp.model.get_objective_expr(), which
        returns the objective expression.
        """
        return self.__model.get_objective_expr()
    
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
        return self.__model.reduced_costs( dvars )
    
    #---------------------------------------------
    
    def chgCoeff(self, constr, dvar, coef=0.0 ):
        """
        Change the coefficient of a single variable dvar 
        in the linear constraint constr to value coef.
        
        Parameters
        ----------
        constr : docplex.mp.constr.LinearConstraint
            The constraint to be adjusted
        dvar : docplex.mp.dvar
            The decision variable where to change
            the coefficient.
        coef : int or float
            The value of the (new) coefficient.
        """
        constr.lhs.set_coefficient(dvar,coef)
        
    #---------------------------------------------
    
    def chgCoeffs(self, constr, coeffs ):
        """
        Changes coefficients of a number of variables
        in the linear constraint constr.
        
        Parameters
        ----------
        coeffs : iterable/sequence of pairs
             iterable/sequence of variable-coefficient pairs
        constr : docplex.mp.constr.LinearConstraint
            The constraint to be adjusted
        """
        for d,c in coeffs: constr.lhs.set_coefficient(d,c)
    
    #---------------------------------------------
    
    def export_model ( self, lp_file ):
        """
        Exports the model to a model file of type "lp". 
        
        Parameters
        ----------
        lp_file : str    
            Name of the file where to write the model.
        """
        self.__model.export_as_lp( lp_file )
    
    #---------------------------------------------
    
    def update(self):
        """
        Update the model. Only relevant for GuRoBi and
        is here just a dummpy.
        """
        pass 
        
    #---------------------------------------------
    
    def end ( self ):
        """
        Clean up the Cplex IP model by calling its "end" method.
        """
        if not self.__model is None: self.__model.end()
        self.__model = None
    
    #---------------------------------------------
    
    def terminate(self):
        """
        Terminate computations from a callback.
        """
        self.__context.abort()
    
    #---------------------------------------------
    
    @property 
    def mipModel(self):
        """
        Return the instance of the docplex.mp.model Model class.
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
        return self.__model.solve_details.best_bound
    
    @property
    def runtime(self):
        """Return computation time used to solve the model"""
        return self.__model.solve_details.time
    
    @property 
    def dettime(self):
        """Return the deterministic time. For Cplex, this
        is the total amout of CPU ticks"""
        return self.__model.get_cplex().get_dettime( )
    
    @property 
    def status(self):
        """Return model's solution status"""
        return self.__model.solve_details.status
    
    @property 
    def nodeCount(self):
        """Return the NodeCount, i.e. the number of branch-and-cut 
        nodes explored in Cplex's most recent optimization"""
        return self.__model.solve_details.nb_nodes_processed
    
    @property 
    def infinity(self):
        """docplex.mp.model.infinity"""
        return self.__model.infinity
    
    @property 
    def MIPGapAbs(self):
        """Return parameter mip.tolerances.absmipgap"""
        return self.__model.parameters.mip.tolerances.absmipgap()
    
    @MIPGapAbs.setter 
    def MIPGapAbs(self, value ):
        """Set parameter mip.tolerances.absmipgap to value"""
        if value >= 0.0:
            self.__model.parameters.mip.tolerances.absmipgap(value)
    
    @property 
    def MIPGap(self):
        """Return parameter mip.tolerances.mipgap"""
        return self.__model.parameters.mip.tolerances.mipgap()
    
    @MIPGap.setter 
    def MIPGap(self,value):
        """Set parameter mip.tolerances.mipgap to value"""
        if value >= 0.0:
            self.__model.parameters.mip.tolerances.mipgap(value)
    
    @property 
    def IntFeasTol(self):
        """Return parameter mip.tolerance.integrality"""
        return self.__model.parameters.mip.tolerances.integrality()
    
    @IntFeasTol.setter 
    def IntFeasTol(self, value):
        """Set parameter mip.tolerance.integrality to value"""
        if 0 <= value < 1: 
            self.__model.parameters.mip.tolerances.integrality(value)
    
    @property
    def FeasibilityTol(self):
        """Return parameter simplex.tolerances.feasibility"""
        return self.__model.parameters.simplex.tolerances.feasibility()
    
    @FeasibilityTol.setter 
    def FeasibilityTol(self,value):
        """Set parameter simplex.tolerances.feasibility"""
        if 0 <= value < 1:
            self.__model.parameters.simplex.tolerances.feasibility(value)
    
    @property 
    def MIQCPMethod(self):
        """Returns value of parameter mip.strategy.miqcpstrat"""
        return self.__model.parameters.mip.strategy.miqcpstrat()
    
    @MIQCPMethod.setter
    def MIQCPMethod(self, strategy ):
        """Sets parameter mip.strategy.miqcpstrat"""
        self.__model.parameters.mip.strategy.miqcpstrat(strategy)
        
    @property 
    def timelimit(self):
        """Time limit for the MIP solver"""
        return self.__model.parameters.timelimit  
    
    @timelimit.setter 
    def timelimit(self, value):
        """Set the time limit for the MIP solver"""
        if value > 0.0: 
            self.__model.parameters.timelimit(value)   
        else: 
            self.__model.parameters.timelimit(1.0E75)
    @property 
    def nodelimit(self):
        """Node limit for the MIP solver"""
        return self.__model.parameters.mip.limits.nodes()  
    
    @nodelimit.setter 
    def nodelimit(self, value):
        """Set the time limit for the MIP solver"""
        if value > 0: 
            self.__model.parameters.mip.limits.nodes(value)
        else:
            self.__model.parameters.mip.limits.nodes(9223372036800000000)   
    
    @property 
    def upper_cutoff( self ):
        """Return upper cutoff value."""
        return self.__model.parameters.mip.tolerances.uppercutoff
    
    @upper_cutoff.setter 
    def upper_cutoff( self, value ):
        """Set an upper cutoff value."""
        self.__model.parameters.mip.tolerances.uppercutoff(value)
    
    @property 
    def lower_cutoff( self ):
        """Return lower cutoff value."""
        return self.__model.parameters.mip.tolerances.lowercutoff
    
    @lower_cutoff.setter 
    def lower_cutoff( self, value ):
        """Set a lower cutoff value."""
        self.__model.parameters.mip.tolerances.lowercutoff(value)
        
    @property 
    def lowerObjStop(self):
        """
        Returns docplex.mp.model.parameters.mip.limits.lowerobjstop 
        """
        return self.__model.parameters.mip.limits.lowerobjstop()
    
    @lowerObjStop.setter 
    def lowerObjStop(self, value ):
        """
        Set Cplex parameter mip.limits.lowerobjstop to the given value.
        """
        self.__model.parameters.mip.limits.lowerobjstop(value)
    
    @property
    def log_output(self):
        """Return value of parameter log_output"""
        return self.__log_output 
    
    @log_output.setter 
    def log_output(self, on_off ):
        """Set log_output on (True) or off (False)"""
        self.__log_output = bool(on_off)
        
    @property 
    def mipStrategy(self):
        """Return value of Cplex parameter mip.strategy.search
        (dynamic search switch)"""
        return self.__model.parameters.mip.strategy.search()
    
    @mipStrategy.setter 
    def mipStrategy(self, value ):
        """Set Cplex parameter mip.parameters.strategy to the
        given value of 0 (automatic), 1 (traditional), 2
        (dynamic search)"""
        if value in (0,1,2): self.__model.parameters.mip.strategy.search()
        
    @property 
    def mipEmphasis(self):
        """Return value of Cplex parameter parameters.emphasis.mip."""
        return self.__model.parameters.emphasis.mip()
    
    @mipEmphasis.setter 
    def mipEmphasis(self, value ):
        """Set Cplex parameter parameters.emphasis.mip to value. We do
        not make use of value=4 (HIDDENFEAS)"""
        if value in (0,1,2,3): self.__model.parameters.emphasis.mip(value)
    
    @property 
    def lbHeur(self):
        """Return Cplex parameter parameters.mip.strategy.lbheur (local
        branching switch"""
        return self.__model.parameters.mip.strategy.lbheur()
    
    @lbHeur.setter 
    def lbHeur(self, value ):
        """Switch Cplex's local branching off (0) or on (1)."""
        if value in (0,1): self.__model.parameters.mip.strategy.lbheur(value)
        
    @property 
    def cuts(self):
        """Return value of parameter cuts"""
        return self.__cuts 
               
    @cuts.setter 
    def cuts( self, value ):
        """
        Set strategy for certain cuts. A value of -1 means automatic
        (Cplex decides), 0 means cuts are switched off, 1 for
        moderate generation of certain cuts, 2 for agressive
        cut generation and 3 even activates disjunctive cuts.
        """
        self.__cuts = value  
        if value==0:
            # Switch off cut generation 
            self.__model.parameters.mip.limits.cutsfactor(0)
        elif value < 0:
            # Automatic cut generation
            self.__model.parameters.mip.limits.cutsfactor(-1)
            self.__model.parameters.mip.cuts.covers.set(0)
            self.__model.parameters.mip.cuts.gubcovers.set(0)
            self.__model.parameters.mip.cuts.flowcovers.set(0)
            self.__model.parameters.mip.cuts.mircut.set(0)
            self.__model.parameters.mip.cuts.gomory.set(0)
            self.__model.parameters.mip.cuts.implied.set(0)
            self.__model.parameters.mip.cuts.liftproj.set(0)
            self.__model.parameters.mip.cuts.disjunctive.set(0)
        elif value==1:
            # Moderate generation of these cuts
            self.__model.parameters.mip.limits.cutsfactor(-1)
            self.__model.parameters.mip.cuts.covers.set(1)
            self.__model.parameters.mip.cuts.gubcovers.set(1)
            self.__model.parameters.mip.cuts.flowcovers.set(1)
            self.__model.parameters.mip.cuts.mircut.set(1)
            self.__model.parameters.mip.cuts.gomory.set(1)
            self.__model.parameters.mip.cuts.implied.set(1)
            self.__model.parameters.mip.cuts.liftproj.set(0)
            self.__model.parameters.mip.cuts.disjunctive.set(0)
        elif value==2: 
            # Aggressive generation of these cuts
            self.__model.parameters.mip.limits.cutsfactor(-1)
            self.__model.parameters.mip.cuts.covers.set(3)
            self.__model.parameters.mip.cuts.gubcovers.set(2)
            self.__model.parameters.mip.cuts.flowcovers.set(2)
            self.__model.parameters.mip.cuts.mircut.set(2)
            self.__model.parameters.mip.cuts.gomory.set(2)
            self.__model.parameters.mip.cuts.implied.set(2)
            self.__model.parameters.mip.cuts.liftproj.set(3)
            self.__model.parameters.mip.cuts.disjunctive.set(0)
        elif value==3:
            # Even use disjunctive cuts
            self.__model.parameters.mip.limits.cutsfactor(-1)
            self.__model.parameters.mip.cuts.covers.set(3)
            self.__model.parameters.mip.cuts.gubcovers.set(2)
            self.__model.parameters.mip.cuts.flowcovers.set(2)
            self.__model.parameters.mip.cuts.mircut.set(2)
            self.__model.parameters.mip.cuts.gomory.set(2)
            self.__model.parameters.mip.cuts.implied.set(2)
            self.__model.parameters.mip.cuts.liftproj.set(3)
            self.__model.parameters.mip.cuts.disjunctive.set(3)
    
    @property
    def doBenders( self ):
        """Return Cplex parameter parameter.benders.strategy"""
        return self.__model.parameters.benders.strategy()
    
    @doBenders.setter 
    def doBenders(self, value ):
        """Set Cplex parameter parameter.benders.strategy. Note
        that value = -1, switches Benders off."""
        if value in (-1, 0, 1, 2, 3):
            self.__model.parameter.benders.strategy(value)
        
    @property 
    def cb_MIPSOL(self):
        """Return the constant cplex.callbacks.Context.id.candidate"""
        return cplex.callbacks.Context.id.candidate

    @property 
    def context(self):
        """Return the current instance of cplex.callbacks.Context"""
        return self.__context
    
    @context.setter 
    def context(self, cntxt ):
        """Keep a handle to the Cplex's callback context instance."""
        self.__context = cntxt
        
    @property 
    def id_relaxation(self):
        """
        Return cplex.callbacks.Context.id.relaxation
        """
        return cplex.callbacks.Context.id.relaxation
    
    @property
    def candidate_objective(self):
        """Return objective value of a candidate solution"""
        return self.__context.get_candidate_objective()
    
    @property
    def incument_objective(self):   
        """Return incumbent objective value"""
        return self.__context.get_incumbent_objective() 

    