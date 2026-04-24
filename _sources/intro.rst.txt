Introduction
============
**pyloa** (python location optimization algorithms) provides a number of methods
for solving location problems in the plane and on a network as well as discrete facility location problems. 

Some of the implemented solution methods require the availability of a mixed integer programming solver. The time being, 
either GuRoBi (via gurobipy) or Cplex (via docplex) can be used to this end. 

Planar location
===============
The sub-package *pyloa.plane* provides the following methods for solving location
problems in the Euclidian plane:

* Fermat-Weber problem using Weiszfeld's, Ostresh's and Drezner's algorithms,
* Planar 1-center problem using Elzinga and Hearn's, Welzl's, and Charalambous' algorithms as well as a primal-dual method and a second order cone formulation.
* Multi-source Weber problem using Cooper's location-allocation method, the p-median heuristic, a second order cone mixed integer programming formulation, a simple variable neighbourhood search, and column generation. Implemented is also Ostresh's method for solving 2-Weber problems exactly.

Data input
----------
Data input can be provided via a text/csv file. The file should at least provide columns for the two coordinates of each customer point. If available, longitude and latitude data can be provided instead. Optional further data include the customer's positive weight, and an id/name/address of a customer point. A very simple example of a data file is the following::

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

Additionally, `tsplib95 <https://tsplib95.readthedocs.io/en/stable/>`_ data files can be used for data input (just coordinates, weights are then assumed to equal 1).

Usage
-----
The following Python snippet can be used to solve a planar location problem::

    from pyloa.plane.planesolver import PlaneSolver

    # DataFile path and name of input data file
    problem = PlaneSolver(DataFile) 

    # Solve Fermat-Weber problem
    problem.solve() 

    # Solve multi-source Weber problem with 7 facility locations
    problem.solve( p = 7 ) 

    # Solve 1-center problem in plane
    problem.solve( minisum=False )

    # Solve p-center problem in plane as a 2nd order MIP
    problem.solve( p=7, minisum=False, method = 'MIPQ' )

    # Solve p-center problem in plane using location-allocation and VNS
    problem.solve( p=7, minisum=False, method = 'VNS' )
    
    # Retrieve solution information
    problem.facilities            # Array of the facilities' coordinates
    problem.assigned              # Nearest facility (index) for each customer point
    problem.max_distance          # Largest unweighted distance to nearest facility
    problem.max_weighted_distance # Largest weighted distance to nearest facility
    problem.distance              # Sum of distances to nearest facility
    problem.weighted_distance     # Sum of weighted distance to nearest facility


Network location
================
The sub-package *pyloa.net* provides methods for solving location problems on a network. Currently, methods for
solving the following network problems are implemented.

* Maximal covering location problem by means of

  - a simple greedy heuristic, 
  - an integer or mixed integer programming solver, and
  - a Lagrangian heuristic based on subgradient optimization.

* p-median problem using

  - a MIP solver,
  - a greedy method, or 
  - a Lagrangian heuristic based on subgradient optimization.

* p-center problem using

  - a MIP solver, or
  - the method by Elloumi, Labbe and Pochet.

Data input
----------
Data can be provided via a text file of same structure as those from `Beasley's OR library <https://people.brunel.ac.uk/~mastjjb/jeb/info.html>`_. It is also
possible to directly load problem instances from the OR library's web side. 

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
    
    # Solve a p-median problem
    problem.pmp_solve()
    
    # Solve a p-center problem
    problem.pcp_solve()
    
    # Solution information to any of the above can be retrieved using the following
    problem.facilities   # List of open facility nodes
    problem.assigned     # List of facilities to which each customer node is assigned
    problem.wdist        # Total weighted distance of customers to facilities 
    problem.get_coverage # Returns total demand covered within maximal distances
    problem.radius       # Largest weighted distance of a customer to nearest facility

Discrete location
=================

This sub-package is undergoing a number of adjustments and will added later.

