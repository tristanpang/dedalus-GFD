from spectralGFD import *
import logging
import dedalus.public as d3
import matplotlib.pyplot as plt
import json
logger = logging.getLogger(__name__)


class time_PDE(DedalusSolver):
    '''
    Subclass of `DedalusSolver`.

    IMPORTANT: create make_problem manually in subclass.

    Useful methods:
    time_plot = plot PDE at specific times
    animate = create video over all time
    animate_old = old animation method
    solve_problem
    '''
    def __init__(self, Nphi, Nr, initial_func, *,
                 Lr=1, dealias=2,
                 timestepper=d3.SBDF2, stop_sim_time=np.pi/2, timestep=0.1,
                 local=True, save_every=1, save_name=None, scales=1,
                 import_previous=False, variable_name=None, **kwargs):
        # Export vars
        self.Nphi = Nphi
        self.Nr = Nr
        self.initial_func = initial_func
        self.Lr = Lr
        self.dealias = dealias
        self.timestepper = timestepper
        self.stop_sim_time = stop_sim_time
        self.timestep = timestep
        self.local = local
        self.save_every = save_every
        self.save_name = save_name
        self.import_previous = import_previous
        self.variable_name = variable_name
        self.scales = scales
        self.__dict__.update(kwargs)

        # Run
        if import_previous is False:
            self.run(local=local, save_every=save_every, save_name=save_name)
        else:
            with open(f'saves/{save_name}/params.json') as json_file:
                data = json.load(json_file)
                self.__dict__.update(data)

    def solve_problem(self, *, local=True, save_every=None, save_name=None):
        '''Solve PDE with given timestepper
        local = True : outputs to variable q_list
        local = False: outputs to file=filename (default classname + time)
        sim_dt: output every sim_dt time
        '''
        # Import vars
        dist = self.dist
        disk = self.disk
        problem = self.problem
        q = self.q
        t = self.t
        timestep = self.timestep
        stop_sim_time = self.stop_sim_time
        save_every = self.save_every

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if save_name is None:
            save_name = 'Snapshots ' + time_str + ' ' + self.__class__.__name__

        sim_dt = save_every * timestep

        # Solver
        solver = problem.build_solver(self.timestepper)
        solver.stop_sim_time = stop_sim_time

        # Main loop (external)
        if local is False:
            snapshots = solver.evaluator.add_file_handler(f'saves/{save_name}',
                                                          sim_dt=sim_dt)
            snapshots.add_tasks(solver.state, layout='g', scales=self.scales)

            while solver.proceed:
                solver.step(timestep)
                if solver.iteration % 100 == 0:
                    logger.info('Iteration=%i, Time=%e, dt=%e'
                                % (solver.iteration, solver.sim_time,
                                   timestep))
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info('Done!')
            clear_output(wait=True)

            # Export vars
            self.sim_ind_list = range(int(stop_sim_time // sim_dt))
            self.save_name = save_name
            self.execution_time = time_str
            self.execution_end_time = end_time
            self.import_previous = True

            with open(f'saves/{save_name}/params.json', 'w') as file:
                json.dump(self.__dict__, file, default=str)

        # Main loop (local)
        if local is True:
            q.change_scales(scales=self.scales)
            q_list = [np.copy(q['g'])]
            t_list = [solver.sim_time]
            while solver.proceed:
                solver.step(timestep)
                if solver.iteration % 100 == 0:
                    logger.info('Iteration=%i, Time=%e, dt=%e'
                                % (solver.iteration, solver.sim_time,
                                   timestep))
                if solver.iteration % save_every == 0:
                    q.change_scales(scales=self.scales)
                    q_list.append(np.copy(q['g']))
                    t_list.append(solver.sim_time)

            logger.info('Done!')
            clear_output(wait=True)

            # Export vars
            self.sim_ind_list = range(int(stop_sim_time // sim_dt))
            self.q_list = q_list
            self.t_list = t_list

    def time_plot(self, plot_t_list=[0, .5, 1], filename=None):
        '''Plots PDE over time. `plot_t_list` is list
        of normalised time in [0,1] to be plotted.'''
        if self.import_previous is True:
            var_name = self.variable_name
            time_plot_file(self.save_name, plot_t_list=plot_t_list,
                           variable_name=var_name, filename=filename)
        else:
            # Import vars
            Nr = self.Nr
            Nphi = self.Nphi
            q_list = self.q_list
            t_list = self.t_list

            # Plotting
            plot_length = len(plot_t_list)
            fig, axs = plt.subplots(1, plot_length, figsize=(20, 6))

            for i, t in enumerate(plot_t_list):
                t_ind = int(t * (len(t_list) - 1))
                self.plot(q_list[t_ind].T, ax=axs[i],
                          title=f'Time = {t_list[t_ind]:.3}')
            fig.tight_layout()
            plt.show()

            if filename is not None:
                fig.savefig(filename+'.png', dpi=fig.dpi,
                            bbox_inches='tight')

    def animate_old(self, pause=0):
        '''Animate PDE over time with global animate function'''
        q_list = self.q_list
        t_list = self.t_list

        def plot_func(k, t):
            self.plot(q_list[k].T, title=f'Time: {t:.3f}')

        animate(plot_func, t_list, pause=pause)

    def animate(self, frames=None):
        '''Animate PDE over time with global animate function'''
        if self.import_previous is True:
            var_name = self.variable_name
            animate_file(self.save_name, variable_name=var_name, frames=frames)
        else:
            q_list = self.q_list
            t_list = self.t_list
            t_steps = len(t_list)

            if frames is None:
                frames = range(t_steps)

            def plot_func(k):
                self.plot(q_list[k].T, ax=ax[0], title=
                          f'Time: {t_list[k]:.3f}', cax=ax[1])

            fig, ax = plt.subplots(1, 3, width_ratios=[10, 1, 1],
                                   figsize=(6, 5))
            ax[2].axis('off')
            ani2 = matplotlib.animation.FuncAnimation(fig, plot_func,
                                                      frames=frames)
            plt.close()

            video = HTML(ani2.to_jshtml())
            display(video)