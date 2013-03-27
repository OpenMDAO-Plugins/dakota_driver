from numpy import array

from dakota import DAKOTAInput, run_dakota

from openmdao.main.datatypes.api import Float, Int, List
from openmdao.main.driver import Driver
from openmdao.main.driver_uses_derivatives import DriverUsesDerivatives
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasconstraints import HasIneqConstraints
from openmdao.main.hasobjective import HasObjective, HasObjectives
from openmdao.main.interfaces import IHasParameters, IHasIneqConstraints, \
                                     IHasObjective, IHasObjectives, \
                                     IOptimizer, implements
from openmdao.util.decorators import add_delegate

__all__ = ['DAKOTAoptimizer', 'DAKOTAstudy']


class DAKOTAmixin(object):
    """ Mixin for common DAKOTA operations. """

    def __init__(self):
        self.input = DAKOTAInput()

    def run_dakota(self):
        """
        Call DAKOTA, providing self as data.
        DAKOTA will then call our :meth:`dakota_callback`.
        """
        infile = self.get_pathname() + '.in'
        self.input.write_input(infile)
        run_dakota(infile, data=self)

    def set_variables(self, need_start):
        """ Set :class:`DAKOTAInput` `variables` section. """
        parameters = self.get_parameters()
        if not parameters:
            self.raise_exception('No parameters, run aborted')

        initial = []
        lbounds = []
        ubounds = []
        names   = []
        for name, param in parameters.items():
            if param.start is None:
                initial.append(str(param.low + (param.high-param.low)/2.))
            else:
                initial.append(str(param.start))
            lbounds.append(str(param.low))
            ubounds.append(str(param.high))
            names.append("%r" % name)

        self.input.variables = []
        self.input.variables.append(
            'continuous_design = %s' % len(parameters))
        if need_start:
            self.input.variables.append(
                '    initial_point %s' % ' '.join(initial))
        self.input.variables.extend([
            '    lower_bounds  %s' % ' '.join(lbounds),
            '    upper_bounds  %s' % ' '.join(ubounds),
            '    descriptors   %s' % ' '.join(names)
        ])


@add_delegate(HasParameters, HasIneqConstraints, HasObjective)
class DAKOTAoptimizer(DriverUsesDerivatives, DAKOTAmixin):
    """ Optimizer using DAKOTA Python interface. """

    implements(IHasParameters, IHasIneqConstraints, IHasObjective, IOptimizer)

    max_iterations = Int(100, iotype='in', low=1)
    convergence_tolerance = Float(1.e-5, iotype='in', low=1.e-10)

    def __init__(self):
        DriverUsesDerivatives.__init__(self)
        DAKOTAmixin.__init__(self)

    def execute(self):
        """ Write DAKOTA input and run. """
        objectives = self.get_objectives()
        if not objectives:
            self.raise_exception('No objective, run aborted')

        self.input.method = [
            'conmin_frcg',
            '    max_iterations = %s' % self.max_iterations,
            '    convergence_tolerance = %s' % self.convergence_tolerance,
        ]
        self.set_variables(need_start=True)
        self.input.responses = [
            'objective_functions = %s' % len(objectives),
            'numerical_gradients',
            '    method_source dakota',
            '    interval_type forward',
            '    fd_gradient_step_size = 1.e-5',
            '    no_hessians',
        ]
        self.run_dakota()

    def dakota_callback(self, **kwargs):
        """ Return repsonses from parameters. """
        cv = kwargs['cv']    # Continuous variables.
        asv = kwargs['asv']  # What to return (bit1=f, bit2=df, bit3=d^2f)
        self._logger.debug('cv %s', cv)
        self._logger.debug('asv %s', asv)

        self.set_parameters(cv)
        self.run_iteration()

# TODO: support full asv.
        val = self.eval_objective()
        retval = dict(fns=array([val]))
        self._logger.debug('returning %s', retval)
        return retval


@add_delegate(HasParameters, HasObjectives)
class DAKOTAstudy(Driver, DAKOTAmixin):
    """ Parameter study using DAKOTA Python interface. """

    implements(IHasParameters, IHasObjectives)

    partitions = List(Int, io_type='in', low=1)

    def __init__(self):
        Driver.__init__(self)
        DAKOTAmixin.__init__(self)

    def execute(self):
        """ Write DAKOTA input and run. """
        parameters = self.get_parameters()
        partitions = [str(partition) for partition in self.partitions]
        if len(partitions) != len(parameters):
            self.raise_exception('#partitions %s != #parameters (%s)'
                                 % (len(partitions), len(parameters)),
                                 ValueError)

        objectives = self.get_objectives()
        if not objectives:
            self.raise_exception('No objectives, run aborted')

        self.input.method = [
            'multidim_parameter_study',
            '    partitions = %s' % ' '.join(partitions)
        ]
        self.set_variables(need_start=False)
        self.run_dakota()

    def dakota_callback(self, **kwargs):
        """ Return repsonses from parameters. """
        cv = kwargs['cv']    # Continuous variables.
        self._logger.debug('cv %s', cv)

        self.set_parameters(cv)
        self.run_iteration()

        vals = self.eval_objectives()
        retval = dict(fns=array(vals))
        self._logger.debug('returning %s', retval)
        return retval

