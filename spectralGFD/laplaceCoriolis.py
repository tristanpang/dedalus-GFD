from spectralGFD import *
from IPython.display import clear_output

class Lap_Cor(DedalusSolver):
    '''
    Subclass of `DedalusSolver`. Adds methods
    `make_problem` and `solve_problem`.

    Solve Laplace Equation with coriolis on Unit Circle
        lap(u) - u/Ld^2 = q;
        u(r=1) = 0, BC.

    Input:
    Ld = coriolis effect
    Nphi, Nr = Grid spacing
    q_func(phi, r, Ld) = q

    'ug' outputs np.array. 'u' outputs dedalus object

    Returns phi, r, u
    '''
    def __init__(self, Nphi, Nr, q_func, *, Ld=np.inf, Lr=1, dealias=1):
        # Export vars
        self.Nphi = Nphi
        self.Nr = Nr
        self.q_func = q_func
        self.Ld = Ld
        self.Lr = Lr
        self.dealias = dealias

        # Run
        self.run()

    def make_problem(self):
        '''Make problem with Laplace equation'''
        # Import vars
        q_func = self.q_func
        Ld = self.Ld
        dist = self.dist
        disk = self.disk
        edge = self.edge
        coords = self.coords
        u = self.u
        Lr = self.Lr

        # Forcing
        phi, r = dist.local_grids(disk)
        q = dist.Field(bases=disk)
        q['g'] = q_func(phi, r, Ld, Lr=Lr)  # as input

        # Tau method
        tau_u = dist.Field(name='tau_u', bases=edge)

        def lift(A):
            lift_basis = disk.derivative_basis()
            return d3.Lift(A, lift_basis, -1)

        # Problem
        problem = d3.LBVP([u, tau_u], namespace=locals())
        problem.add_equation("lap(u) - u/(Ld**2) + lift(tau_u) = q")
        problem.add_equation("u(r=Lr) = 0")

        # Export vars
        self.problem = problem

    def solve_problem(self, local=None, save_every=None, save_name=None):
        '''Solve Laplace equation'''
        # Import vars
        dist = self.dist
        disk = self.disk
        problem = self.problem
        u = self.u

        # Solver
        solver = problem.build_solver()
        solver.solve()

        # Gather global data
        phi, r = dist.local_grids(disk)
        ug = u.allgather_data('g')
        print('Done!')

        clear_output(wait=True)

        # Export vars
        self.u = u
        self.ug = ug
        self.phi = phi
        self.r = r