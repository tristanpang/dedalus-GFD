from spectralGFD import *

class stommel_PDE(time_PDE):
    def make_problem(self):
        '''Make PDE problem:
        Stommel Equation on Unit Circle
        dt(u) + 0.5 * gradperp(r^2) @ grad(u) = Q - r_0 lap(r^2);
        where gradperp(.) = skew(grad(.)) in d3'''
        # Import vars
        dist = self.dist
        disk = self.disk
        edge = self.edge
        coords = self.coords
        Lr = self.Lr
        r0 = self.r0
        F = self.F
        H = self.H
        beta = self.beta
        nu = self.nu
        rho0 = self.rho0
        Q_shift = self.Q_shift

        # Overwrite fields from make_space
        t = dist.Field()

        # Substitutions
        psi = dist.Field(name='psi', bases=disk)
        zeta = dist.Field(name='zeta', bases=disk)

        phi, r = dist.local_grids(disk)
        y = dist.Field(name='y', bases=disk)
        y['g'] = r * np.sin(phi)

        Q = dist.Field(name='Q', bases=disk)
        Q['g'] = (F * np.pi / (rho0 * Lr * H)) *\
                 np.sin(np.pi * (r * np.sin(phi) + Lr*Q_shift) /Lr)


        # Tau method
        tau_zeta = dist.Field(name='tau_zeta', bases=edge)
        tau_psi = dist.Field(name='tau_psi', bases=edge)

        def lift(A, i=-1):
            lift_basis = disk.derivative_basis()
            return d3.Lift(A, lift_basis, i)

        # Problem
        problem = d3.IVP([zeta, psi, tau_zeta, tau_psi], time=t, namespace=locals())
        problem.add_equation("dt(zeta)  + r0 * lap(psi) - nu * lap(zeta) + lift(tau_zeta, -2) \
                             = -skew(grad(psi)) @ grad(zeta + beta * y)  + Q")
        problem.add_equation("lap(psi) - zeta + lift(tau_psi, -1) = 0")
        problem.add_equation("psi(r=Lr) = 0")
        problem.add_equation("zeta(r=Lr) = 0")

        # Initial conditions
        psi_init, zeta_init, t_init = self.initial_func
        zeta['g'] = zeta_init(phi, r)
        psi = self.initial_condition(psi, zeta)
        t['g'] = t_init

        # Export vars
        self.q = psi
        self.psi = psi
        self.zeta = zeta
        self.Q = Q
        self.y = y
        self.tau_zeta = tau_zeta
        self.tau_psi = tau_psi
        self.t = t
        self.problem = problem
        self.variable_name = 'psi'

    def initial_condition(self, psi, zeta):
        '''Set initial conds'''
        # Tau method
        tau = self.dist.Field(name='tau_zeta', bases=self.edge)

        def lift(A):
            lift_basis = self.disk.derivative_basis()
            return d3.Lift(A, lift_basis, -1)

        # problem
        ic_problem = d3.LBVP([psi, tau], namespace=locals())
        ic_problem.add_equation("lap(psi) + lift(tau) = zeta")
        ic_problem.add_equation("psi(r=self.Lr) = 0")

        # Solver
        solver = ic_problem.build_solver()
        solver.solve()
        return psi