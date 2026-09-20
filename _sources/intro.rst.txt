Introduction
============
**pyloa** (:underline:`py`\thon :underline:`l`\ocation :underline:`o`\ptimization :underline:`a`\lgorithms) provides a number of methods for solving location problems in the plane and on a network as well as discrete facility location problems. 

Some of the implemented solution methods require the availability of a mixed integer programming solver. The time being, 
either GuRoBi (via gurobipy) or Cplex (via docplex) can be used to this end. 

Planar location
===============
The sub-package *pyloa.plane* provides methods for solving the following location problems in the Euclidian plane:

* The *Fermat-Weber problem* using Weiszfeld's, Ostresh's and Drezner's algorithms,
* The *planar 1-center problem* using Elzinga and Hearn's, Welzl's, and Charalambous' algorithms as well as a primal-dual method and a second order cone formulation.
* The *multi-source Weber problem* using Cooper's location-allocation method, the p-median heuristic, a second order cone mixed integer programming formulation, a simple variable neighbourhood search, and column generation. Implemented is also Ostresh's method for solving 2-Weber problems exactly.

Data input
----------
Data input can be provided via a text/csv file. The file should at least provide columns for the two coordinates of each customer point. If available, longitude and latitude data can be provided instead. Optional further data include the customers' positive weights, and an id/name/address of a customer point. A very simple example of a data file is the following::

    X;Y;weight
    0;0;1
    5;0;1
    12;6;1
    0;10;1

Another one providing longitude and latitude data is::

    address;lat;long
    Aarhus;56.1496278;10.2134046
    Herning;56.1379757;8.9746623
    Horsens;55.8611696;9.8444774
    Aalborg;57.0462626;9.9215263
    Esbjerg;55.4664892;8.4520751
    Aabenraa;55.0446228;9.4209667
    Sønderborg;54.9089186;9.7897999

It is also possible to just provide address data without any coordinates. It is then tried to find the longitude and latitude of each address by means of the Python package `geopy <https://geopy.readthedocs.io/en/stable/>`_. The addresses should be precise enough in order to avoid ambiguities. A few parameters that geopy uses when sending requests to the geocoder
*Nominatim* can be controlled by means of the function *set_Nominatim_params* implemented in the module *pyloa.plane.parser*. Instead of *Nominatim*, it is also possible to use
`OpenCage's geocoder <https://opencagedata.com>`_. An api-key is required for using this service. Up to 2500 geocoding requests per day are free. Switching to OpenCage's geocoder
can be done via the function *set_geoServer* implemented in module *pyloa.plane.parser*. Moreover, `tsplib95 <https://tsplib95.readthedocs.io/en/stable/>`_ data files can be used for data input (just coordinates, weights are then assumed to equal 1).

Instead of providing the required data via a file, they can also directly be passed to an instance of the class as arguments
when creating the class instance.

Usage
-----
The following Python snippet illustrates how to solve a planar location problem::

    from pyloa.plane.planesolver import PlaneSolver

    # DataFile path and name of input data file
    problem = PlaneSolver(DataFile) 

    # Solve a Fermat-Weber problem
    problem.solve() 

    # Solve a multi-source Weber problem with 7 facility locations
    # by the default method (Location-allocation)
    problem.solve( p = 7 ) 
    
    # Solve a multi-source Weber problem as a quadratically constrained MIP
    problem.solve( p=7, method = 'MIPQ' )

    # Solve a multi-source Weber problem by means of column generation
    problem.solve( p=7, method = 'colgen' )

    # Solve a 1-center problem in plane
    problem.solve( minisum=False )

    # Solve a p-center problem in plane as a quadratically constrained MIP
    problem.solve( p=7, minisum=False, method = 'MIPQ' )

    # Solve a p-center problem in plane using location-allocation and VNS
    problem.solve( p=7, minisum=False, method = 'VNS' )
    
    # Retrieve solution information
    problem.facilities            # Array of the facilities' coordinates
    problem.assigned              # Nearest facility (index) for each customer point
    problem.max_distance          # Largest unweighted distance to nearest facility
    problem.max_weighted_distance # Largest weighted distance to nearest facility
    problem.distance              # Sum of distances to nearest facility
    problem.weighted_distance     # Sum of weighted distance to nearest facility

    # Remark: Let m be the number of customer points, Y the numpy array of the
    # m customer coordinates and w the numpy array of customer weights.
    # Passing data directly to the class works as follows.
    problem = PlaneSolver( Y=Y, w=w )
    
    # For plotting a solution use:
    problem.plot()
    
Remark
------
When geospatial coordinates are available the plotting function above tries to plot the solution using a *folium* map and
calls the default web browser for displaying the html file created by `folium <https://realpython.com/python-folium-web-maps-from-data/>`_.
Depending on the employed browser, it can happen that the browser refuses to display the html file. In this case, either
just another browser to pen the created html file or, for instance, use Python's http server to display this html file on local host's address.

Network location
================
The sub-package *pyloa.net* provides methods for solving location problems on a network. Currently, methods for
solving the following network problems are implemented.

* The *maximal covering location problem* by means of

  - a simple greedy heuristic, 
  - an integer or mixed integer programming solver, and
  - a Lagrangian heuristic based on subgradient optimization.

* The *p-median problem* using

  - a MIP solver,
  - a greedy method, or 
  - a Lagrangian heuristic based on subgradient optimization.

* The *vertex p-center problem* using

  - a MIP solver, or
  - the method by Elloumi, Labbe and Pochet (2004).

Data input
----------
Data can be provided via a text file structured as those from `Beasley's OR library <https://people.brunel.ac.uk/~mastjjb/jeb/info.html>`_. It is also possible to directly load problem instances from the OR library's web side. Data may also be provided
as arguments when creating the class instance.

A small example is the following data of a maximal covering location problem::

    6 11 2
    1 2 8
    1 3 15
    1 4 10
    2 3 12
    2 4 7
    2 5 16
    3 5 9
    3 6 11
    4 5 11
    4 6 17
    5 6 13
    10 10
    8 10
    22 10
    18 10
    7 10
    55 10

The first line above gives the number of nodes, the number of edges and the number of facilities to locate. Thereafter, the
end nodes and length of each edge follow. Finally, the weight and maximal radius of each customer node is given.

Usage
-----
The following Python snippet illustrates how to use the package pyloa.net::

    from pyloa.net.netsolver import NetSolver 
    
    problem = NetSolver(DataFile)
    
    # Solve a maximal covering location problem
    problem.mcl_solve()
    
    # Solve a p-median problem by means of the MIP solver
    problem.pmp_solve()

    # Solve a p-median problem by means of a Lagrangian heuristic
    problem.pmp_solve( method='subgradient')
        
    # Solve a vertex p-center problem
    problem.pcp_solve()
    
    # Solve a vertex p-center problem using Elloumi et al.'s algorithm
    problem.pcp_solve( method='Elloumi' )
    
    # Solution information to any of the above can be retrieved using the following
    problem.facilities   # List of open facility nodes
    problem.assigned     # List of facilities to which each customer node is assigned
    problem.wdist        # Total weighted distance of customers to facilities 
    problem.get_coverage # Returns total demand covered within maximal distances
    problem.radius       # Largest weighted distance of a customer to nearest facility

    # Passing data to the class instance when creating it, works as follows.
    problem = NetSolver( dmat, p=7, w=w, dmax=dmax )
    # Above, dmat is the numpy matrix of distances, p=7 facilities are to open, 
    # w is the numpy array of customer/node weights and dmax the maximal distances to
    # observe in case of a maximal covering location problem. The following is
    # equivalent to the above.
    problem = NetSolver()
    problem.set_data( dmat, p=7, w=w, dmax=dmax )


Discrete location
=================

This sub-package provides methods for solving discrete facility location problems. The module *pyloa.discrete.cflp* provides
methods for solving the *capacitated facility location problem*, module *pyloa.discrete.uflp* contains some additional methods
for the *uncapacitated facility location problem*, and module *pyloa.discrete.tlcflp* provides some methods for *two-stage*
and *two-level (un-)capacitated facility location problems*.

Data input
----------

**Capacitated facility location**

The following file formats can be employed for providing the required data to a CFLP:

1. The format of Beasley's OR-library files (*formt='CFL-OR'*)
2. The format used by Guastaroba and Speranza (2009), (*formt='CFL-G'*)
3. The format used by Avella and Boccia (2009), (*formt*='CFL-AO'*)
4. The format used by Avella et al. (2021), (*formt*='CFL-AN'*)
5. The format used for the Goertz-Klose instances (*formt='CFL-GK'*)

The cost :math:`c_{ji}` is the cost to supply all of customer :math:`i`'s demand from a facility at site :math:`j` in
case of the file formats 'CFL-OR' and 'CFL-GK'; otherwise, :math:`c_{ji}` describes the cost per unit of demand.

A simple example of the last file format is the following::

    [CFLP-PROBLEMFILE]
    Small example of file format 'CFL-GK'

    [CUSTOMERS]
    demand name
    2 cust1
    4 cust2
    2 cust3
    5 cust4
    4 cust5

    [DEPOTS]
    capacity fixcost
    4 15
    10 15
    8 15
    6 15
    5 15
    5 15

    [COSTMATRIX]
    Matrix c(j,i). Rows=Depots
    [MATRIX]
    Dim 6 5
    12 3 54 96 68
    14 83 59 94 60
    18 89 15 69 64
    32 5 24 23 26 
    15 32 67 16 16
    30 29 23 38 15

**Uncapacitated facility location** 

Data can be provided in just the same way as for the CFLP. Additionally, the *simple UflLib file format* can be used (formt='UFL-SIMPLE'). A number of test problem instances from `UflLib <https://resources.mpi-inf.mpg.de/departments/d1/projects/benchmarks/UflLib>`_ show this format. See the link for examples.


**Two-stage capacitated facility location**

With that we mean the problem of finding optimal capacitated depot locations that serve a given customer set and are supplied by capacitated warehouses (or plants) at **given** locations. For providing the data, the file format used in Klose (1999, 2000) can be used. The example below explains how these files are structured::

    [TSCFLP-PROBLEMFILE]
    Example of TSCFLP data file

    [PLANTS]
    capacity name 
    154 Plant1
    10 Plant2
    227 Plant3
    128 Plant4
    152 Plant5

    [DEPOTS]
    capacity fixcost name
    135 1319 Depot1
    110 1132 Depot2
    117 1218 Depot3
    10 352 Depot4
    65 952 Depot5
    23 527 Depot6

    [CUSTOMERS]
    demand name
    14 Customer1
    32 Customer2
    27 Customer3
    35 Customer4
    28 Customer5
    21 Customer6
    14 Customer7
    22 Customer8
    30 Customer9
    10 Customer10

    [FSCOSTMAT]
    c= d_eucli(a,b) * 0.0075. Unit cost to supply depots (rows) from plants (columns)
    [MATRIX]
    Dim 6 5
    4.6466 3.4422 7.0616 4.0093 2.6025
    3.8255 5.1573 6.8134 5.3638 2.6025
    5.6040 0.6983 6.7790 1.6575 2.6025
    1.1908 7.2327 3.9678 6.4532 2.6025
    4.4700 5.6693 2.3761 4.0627 2.6025
    2.5371 7.3197 0.9753 5.9886 2.6025

    [SSCOSTMAT]
    c= d_eucli(a,b) * 0.01. Cost to supply all of a customer's demand (rows) from depots (columns)
    [MATRIX]
    Dim 6 10
    23.2084 28.6521 215.9494 162.3789 118.5469 70.0834 45.2495 175.7449 115.849 11.7346 
    26.5336 52.0922 180.0924 77.9966 50.5158 70.9492 20.3583 169.4459 166.6081 13.5336 
    59.6949 146.3104 250.0528 273.341 209.1637 93.6621 97.0409 170.9021 52.8417 24.769 
    75.242 209.3067 40.6921 137.0687 117.4352 91.5451 90.1284 96.9329 221.0452 31.1538 
    110.1936 297.3463 175.697 340.7924 271.8142 128.3595 153.1188 68.1279 172.8638 44.6386
    104.5793 286.3249 74.4544 267.5287 218.3659 119.7792 136.1598 33.353 222.9709 42.6095 

**Two-level capacitated facility location**

In this case, the locations of the warehouses (or plants) on the lower echelon have to be determined as well. In addition to the capacities, warehouse fixed cost are thus to be supplied as well. Data can be provided using the same format as above, just that the *plant* data need to have an additional column named *fixcost*. Moreover, the file format of the test instances from Fernandes et al. (2014) can be used as well. The first line of such a file provides the number of plant sites, the number of depot sites and the number of customers. The following lines give each single customer’s demand. The next lines provide for each plant site its capacity and fixed cost. Subsequently, comes the matrix of unit cost to forward one unit of demand from the plants (rows) to the depots (columns). The following lines specify capacity and fixed cost of each potential depot site. Finally, the last lines contain, for each depot site (rows), the cost per unit to meet the customers’ (columns) demand from the depot in the respective row.


Usage
-----

**Capacitated facility location**

For solving the CFLP either a MIP solver (Cplex or GuRoBi) or a branch-and-bound method based on Lagrangian relaxation and subgradient optimization can be employed. The MILP model can be based on the problem's strong or weak LP formulation. It is also possible to generate *variable upper bound constraints* as user cuts. The Lagrangian-based branch-and-bound method employs the network simplex method from the `Lemon Graph Library <https://lemon.cs.elte.hu/trac/lemon>`__ (Deszö et al., 2011) for solving the transportation problems that arise when the set of open facilities is given. Moreover, it uses `David Pisinger's C-code <https://hjemmesider.diku.dk/~pisinger/codes.html>`__ of the Combo algorithm (Martello et al., 1999) for solving the binary knapsack problem arising within the Lagrangian subproblem.

The following code snippet illustrates how to use the module::

    from pyloa.discrete.cflp import CFLP

    cflp = CFLP(DataFile)

    # Solve using Cplex or GuRoBi. Default is Cplex if both are installed.
    cflp.solver = 'mip'  # Use the MIP solver.
   
    cflp.mipSolver = 'gurobi' # If GuRoBi is installed 
    cflp.mipSolver = 'cplex'  # If Cplex is installed

    cflp.adVubs  = 0 # Use weak problem formulation
    cflp.addVubs = 1 # Generate variable upper bounds as user cuts
    cflp.addVubs = 2 # Include variable upper bounds in the initial model (recommended)
    
    cflp.nodeLim = 100 # Limit the number of branch-and-cut nodes if desired
    cflp.timeLim = 100 # Limit the computation time in seconds if desired 
    
    cflp.silent = True  # No solver log-output to stdout
    cflp.silent = False # Is the default
    
    cflp.default_options() # Back to default options
    
    cflp.solve( ) # Solve the problem instance
    
    # Output 
    cflp.facilities # List of open facilities (indices)
    cflp.assigned   # Assignment of customers to open facilities (list) if uncapacitated
    cflp.supply     # Product flows supply[(j,i)] from facility j to customer i (if capacitated)
    cflp.fcost      # Total fixed facility cost
    cflp.tcost      # Total supply/transportation cost
    cflp.cost       # self.fcost + self.tcost
    cflp.mip_time   # Solver's computation time in seconds

    # Solve using Lagrangian relaxation based branch-and-bound
    cflp.solver = 'SG'
    cflp.solve()
    # Output is the same as above, except of mip_time 
    cflp.ctime # Computation time in CPU seconds and as wall time
    
    # If Cplex is available, its built-in Benders decomposition can be used as well.
    cflp.solver = 'mip'
    cflp.mipSolver = 'cplex'
    cflp.addVubs = 2 
    cflp.useBenders = 1
    cflp.solve()

    # Passing the data directly to a class instance can be done as follows.
    cflp = CFLP(unitCost=False, d=d, s=s, f=f, c=c )
    # Above, d is the numpy array of customer demands, s the array of depot capacities,
    # f the fixed cost array and c[j,i] is the cost of supplying all of customer i's
    # demand from a facility at site j. If unitCost=True, it is however assumed that
    # c[j,i] is the cost per unit of demand. The above is equivalent to
    cflp = CFLP()
    cfl.set_data( f, c, d, s, unitCost=False ) 

**Uncapacitated facility location**

Uncapacitated facility location problems can be solved just as a CFLP using the module *pyloa.discrete.cflp*. The module *pyloa.discrete.uflp* just adds a few additional procedures:

1. A Lagrangian relaxation based heuristic employing subgradient optimization.
2. The classical Add and Drop procedures.
3. The classical Interchange (or Swap) procedure.

The following code snippet illustrates how to use the module:: 

    from pyloa.discrete.uflp import UFLP 
    
    # Read data from a CFLP ORlib file 
    ufl = UFLP( DataFile )
    
    # Read data from using the simple UflLib formate
    ufl = UFLP( DataFile, formt='UFL-SIMPLE')
    
    # Read data from the other CFLP file formats mentioned above
    formt = 'CFL-GK' # or, alternatively 'CFL-G', 'CFL-AO', 'CFL-AN'
    ufl = UFLP( DataFile, formt=formt )
    
    # Solve the problem
    ufl.solve( method='LR'  ) # Lagrangian heuristic
    ufl.solve( method='ADD' ) # Add heuristic
    ufl.solve( method='DROP') # Drop heuristic
    ufl.solve( method='DI'  ) # Drop followed by Interchange
    ufl.solve( method='AI'  ) # Add followed by Interchange
    
    # Output
    uflp.facilities # List of open facilities (indices)
    ufl.assigned   # Assignment of customers to open facilities (list)
    ufl.fcost      # Total fixed facility cost
    ufl.tcost      # Total supply/transportation cost
    ufl.cost       # self.fcost + self.tcost
    ufl.itr        # iterations
    ufl.ctime      # Computation time in CPU seconds and as wall time

    # As in case of the CFLP, data can also be passed as arguments when
    # creating the class. Below, f is the numpy array of fixed facility cost
    # and c the numpy matrix such that c[j,i] is the cost to serve customer
    # i from facility j.
    ufl = UFLP( f=f, c=c )
    # In case that c is a matrix of unit cost (per unit of demand) use
    ufl = UFLP( f=f, c=c, d=d ) # where d is the array of customer demands.
    # Equivalent to the above is the following.
    ufl = UFLP()
    ufl.set_data( f=f, c=c, d=d )

**Two-stage and two-level facility location** 

The module *pyloa.discrete.tlcflp* provides routines to solve two-stage and two-level facility location problems heuristically as well as by a MIP solver (Cplex or GuRoBi). In case of a two-level problem, either the problem's multi-commodity formulation or a flow formulation can be employed. The first one uses three-indexed variables :math:`x_{ijk}` for the share of customer's :math:`k` demand met from plant :math:`i` and depot :math:`j`, whereas the second formulation uses variables :math:`v_{ij}` for the amount of product shipped from plant :math:`i` to depot :math:`j` and variables :math:`x_{jk}` for the share of customer :math:`k`'s demand met from depot :math:`j`. As in case of the CFLP, for both formulations, variable upper bounds may either be ignored, included as user cuts, or included in the initial model formulation. Note that the multi-commodity formulation actually only makes sense if these constraints are not ignored.

The following code snippet shows how to use the module for solving an instance of the TSCFLP (fixed plant sites with given capacities,
depot sites need to be located)::

    from pyloa.discrete.tlcflp import TLCFLP
    
    # T111.TCF contains data of small instance with 5 plant sites, 10 depot sites, 25 customers 
    tscfl = TLCFLP('T111.TCF',formt='TSCFL')
    
    # Default options
    tscfl.default_options()
    
    # Solve using the MIP solver (Cplex is default if present)
    tscfl.solve()
    
    # Treatment of variable upper bounds
    tscfl.addVubs = 0 # ignore them
    tscfl.addVubs = 1 # include as user cuts
    tscfl.addVubs = 2 # include all in the initial model (this is the default)
    
    # Solution attributes
    tscfl.cost  # Total cost of the solution
    tscfl.fcost # Total fixed (depot) cost
    tscfl.tcost # Tuple of cost to supply depots from plants and customers from depts
    tscfl.facilities # List of open depot sites
    tscfl.supply[0]  # Dictionary of supplies from plants to depots
    tscfl.supply[1]  # Dictionary of supplies from depots to customers 
                     # (as share of a customer's demand)
    tscfl.mip_time   # Computation time used by the MIP solver
    
    # Benders' decomposition may work better than the ordinary MIP solver 
    # but requires the time being Cplex's built-in Benders' method.
    tscfl.useBenders = 1
    tscfl.solve()
    
    # Remark: For suppressing the solver sending log-output to stdout use
    tscfl.silent = True
    
    # The module also includes a Lagrangian heuristic. The heuristic relaxes the plant
    # capacities in a Lagrangian manner. The resulting Lagrangian subproblem can be
    # reduced to a CFLP. Solving this gives a set of open depots. The min-cost network
    # flow problem is then solved and the plant vertices' node potentials (dual variables)
    # used as Lagrangian multipliers for the next iteration. The procedure continues
    # until a maximal number of iterations is reached, a set of open depots is generated
    # a second time, or optimality should be established. Note that this procedure
    # resembles the subproblem phase of the Van Roy's Cross Decomposition method.
    tscfl.heuristic( min_iter=3, max_iter=5 ) # Do at least 3 and at most 5 iterations
 
    # All output is the same as usual. Except the following:
    tscfl.ctime # CPU time and walltime in seconds used by the heuristic
    tscfl.bound # The obtained lower bound on the optimal objective value  
    
    # Above, the Lagrangian subproblem is solved by the MIP solver. Instead, a Lagrangian
    # relaxation based branch-and-bound for the CFLP can be used.
    tscfl.cflp_solver = 'sg'
    tscfl.heuristic( min_iter=3, max_iter=5 )
    
    
In the similar way, two-level capacitated facility location problems can be addressed. In this case, however, we have the
additional option to employ the multi-commodity formulation, which shows a stronger LP relaxation but a larger number
of variables. Variable upper bound constraints should be included, but perhaps just as user cuts. Also applying
Benders' decomposition might give better results. Consider the data file T111.TCF used above for the TSCFLP and assume that
we include fixed cost data to the plant data (cf. the section on data input). Let then TL111.TCF be the adjusted data
file describing an instance of the TLCFLP. We may then use pyloa's module *tlcflp* as illustrated below::

    from pyloa.discrete.tlcflp import TLCFLP
    
    # Create the class instance from the data. Use the problem's flow formulation.
    tlcfl = TLCFLP('TL111.TCF', formt='TLCFL-AK' )
    
    # Solve the problem instance using default options. 
    # In particular, variable upper bounds are included.
    tlcfl.solve()
    
    # Solution attributes:
    tlcfl.cost       # Total cost of the solution
    tlcfl.fcost      # Fixed cost of the open plant and open depot sites 
    tlcfl.tcost      # Cost of supplying open depots from open plants
                     # and customers from the open depots
    tlcfl.facilities # List of open plant and open depot sites
    tlcfl.supply[0]  # Dictionary of the amount of product shipped from plant to depot sites
    tlcfl.supply[1]  # Dictionary of shipments from depots to customers expressed 
                     # as shares of customer demands.
    tlcfl.mip_time   # Computation time in seconds used by the solver
    
    # The multi-commodity formulation might at times perform better, in particular, 
    # if variable upper bound constraints are generated as user cuts.
    tlcfl = TLCFLP('TL111.TCF', formt='TLCFL-AK', mcf=True )
    tlcfl.addVubs=1
    tlcf.solve()
    # All outputs are the same as above, just that the supply dictionary is now a dictionary of 
    # flows from plants via depots to customers. Flows are expressed as shares of the 
    # customers' demand.
    tlcfl.supply
    
    # If Cplex is the MIP solver, we may also use its built-in Benders decomposition
    tlcfl.useBenders=1 
    tlcfl.addVubs=2
    
    # The similar can of course be done when using the flow formulation
    tlcfl = TLCFLP('TL111.TCF', formt='TLCFL-AK' )
    tlcfl.useBenders = 1
    tlcfl.solve()

    # A Lagrangian heuristic is also available for approximately solving the TLCFLP. The heuristic
    # requires the flow formulation and is based on relaxing the flow conversation constraints in
    # a Lagrangian manner. The Lagrangian subproblem decomposes into a CFLP for determining the 
    # set of open depots and a series of continuous knapsack and a single binary knapsack problem
    # for determining the open plant sites. Lagrangian multipliers are then, similar as done for
    # the TSCFLP, obtained from the node potentials after solving the min-cost network flow
    # problem for given sets of open plant and depot sites.
    tlcfl.heuristic( min_iter=3, max_iter=5 ) # at least 3 and at most 5 iterations
    tlcfl.bound # Obtained lower bound on the optimal objective function value
    tlcfl.ctime # CPU and walltime in seconds     


References
==========

Avella P, Boccia M (2009) A cutting plane algorithm for the capacitated facility location problem. Computational Optimization and Applications 43:39–65.

Avella P, Boccia M, Mattia S, Rossi F (2021) Weak flow cover inequalities for the capacitated facility location problem. European Journal of Operational Research 289:485–494.

Charalambous C (1982) Extension of the Elzinga-Hearn algorithm to the weighted case. Operations Research 30:591–594.

Cooper, L (1963) Location-allocation problems. Operations Research 11:331-343.

Drezner Z (1992) A note on the Weber location problem. Annals of Operations Research 40:153–161.

Dezsö B, Jüttner A, Kovács P (2011) LEMON – an Open Source C++ Graph Template Library. Electronic Notes in Theoretical Computer Science 264:23-45.

Elloumi S, Labbé M, Pochet Y (2004) A new formulation and resolution method for the p-center problem. INFORMS Journal on Computing 16:84–94

Fernandes DRM, Rocha C, Aloise D, Ribeiro GM, Santos EM, Silva A (2014) A simple and effective genetic algorithm for the two-stage capacitated facility location problem. Computers & Industrial Engineering 75:200–208. 

Görtz S, Klose A (2012) A simple but usually fast branch-and-bound algorithm for the capacitated facility location problem. INFORMS Journal on Computing 24:597–610.

Guastaroba G, Speranza MG (2009) Kernel search for the capacitated facility location problem. Journal of Heuristics 18:877–917.

Klose A (1999) An LP-based heuristic for two-stage capacitated facility location problems. The Journal of the Operational Research Society 50:157–166.

Klose A (2000) A Lagrangean relax-and-cut approach for the two-stage capacitated facility location problem. European Journal of Operational Research 126:185–198.

Klose A (2026) Optimisation Models and Methods for Location Planning -- with Implementations in Python. Springer Nature, Graduate Texts in Operations Research. 

Martello S, Pisinger D, Toth P (1999) Dynamic programming and strong bounds for the 0-1 knapsack problem. Management Science 45:414–424.

Ostresh LM (1978) On the convergence of a class of iterative methods for solving the Weber location problem. Operations Research 26:597–609.

Ostresh LM (1975) An efficient algorithm for solving the two center location-allocation problem. Journal of Regional Science 15:209–216.

Van Roy TJ (1986) A cross decomposition algorithm for capacitated facility location. Operations Research 34:145–163.

Welzl E (1991) Smallest enclosing disks (balls and ellipsoids). In: Maurer H (ed) New Results and New Trends in Computer Science, Lecture Notes in Computer Science, vol 555, pp 359–370.
