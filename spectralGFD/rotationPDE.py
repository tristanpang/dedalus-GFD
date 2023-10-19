from spectralGFD import *

class rotation_PDE(time_PDE):
    def make_problem(self):
        '''Make PDE problem:
        Advection Equation on Unit Circle
        dt(psi) + 0.5 * gradperp(r^2) @ grad(psi) = 0;
        where gradperp(.) = -skew(grad(.)) in d3'''
        # Import vars
        dist = self.dist
        disk = self.disk
        edge = self.edge
        coords = self.coords
        Lr = self.Lr

        # Overwrite fields from make_space
        psi = dist.Field(name='psi', bases=disk)
        t = dist.Field()

        # Substitutions
        phi, r = dist.local_grids(disk)

        u = dist.VectorField(coords, bases=disk)
        u['g'][0] = r
        u['g'][1] = 0

        # Problem
        problem = d3.IVP([psi], time=t, namespace=locals())
        problem.add_equation("dt(psi) = - u @ grad(psi)")

        # Initial conditions
        psi['g'] = self.initial_func(phi, r)

        # Export vars
        self.q = psi
        self.t = t
        self.problem = problem
        self.variable_name = 'psi'


