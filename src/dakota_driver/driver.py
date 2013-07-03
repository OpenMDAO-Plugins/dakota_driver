from numpy import array

from dakota import DakotaInput, run_dakota

from openmdao.main.datatypes.api import Bool, Enum, Float, Int, List, Str
from openmdao.main.driver import Driver
from openmdao.main.driver_uses_derivatives import DriverUsesDerivatives
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasconstraints import HasIneqConstraints
from openmdao.main.hasobjective import HasObjective, HasObjectives
from openmdao.main.interfaces import IHasParameters, IHasIneqConstraints, \
                                     IHasObjective, IHasObjectives, \
                                     IOptimizer, implements
from openmdao.util.decorators import add_delegate

__all__ = ['DakotaOptimizer', 'DakotaMultidimStudy', 'DakotaVectorStudy']


class DakotaMixin(object):
    """
    Mixin for common DAKOTA operations, adds :class:`DakotaInput` instance.
    The ``method`` and ``responses`` sections of `input` must be set
    directly.  :meth:`set_variables` is typically used to set the ``variables``
    section.
    """

    def __init__(self):
        # Set baseline input, don't touch 'interface'.
        self.input = DakotaInput(strategy=['single_method'],
                                 method=[],
                                 model=['single'],
                                 variables=[],
                                 responses=[])

        self.add('stdout', Str('', iotype='in',
                               desc='DAKOTA stdout filename'))
        self.add('stderr', Str('', iotype='in',
                               desc='DAKOTA stderr filename'))
        self.add('tabular_graphics_data',
                 Bool(iotype='in',
                      desc="Record evaluations to 'dakota_tabular.dat'"))

    def set_variables(self, need_start):
        """ Set :class:`DakotaInput` ``variables`` section. """
        parameters = self.get_parameters()

        initial = []
        lbounds = []
        ubounds = []
        names   = []
        for name, param in parameters.items():
            start = param.evaluate() if param.start is None else param.start
            initial.append(str(start))
            lbounds.append(str(param.low))
            ubounds.append(str(param.high))
            names.append("%r" % name)

        self.input.variables = [
            'continuous_design = %s' % len(parameters)]
        if need_start:
            self.input.variables.append(
                '    initial_point %s' % ' '.join(initial))
        self.input.variables.extend([
            '    lower_bounds  %s' % ' '.join(lbounds),
            '    upper_bounds  %s' % ' '.join(ubounds),
            '    descriptors   %s' % ' '.join(names)
        ])

    def run_dakota(self):
        """
        Call DAKOTA, providing self as data, after enabling or disabling
        tabular graphics data in the ``strategy`` section.
        DAKOTA will then call our :meth:`dakota_callback` during the run.
        """
        if not self.input.method:
            self.raise_exception('Method not set', ValueError)
        if not self.input.variables:
            self.raise_exception('Variables not set', ValueError)
        if not self.input.responses:
            self.raise_exception('Responses not set', ValueError)

        for i, line in enumerate(self.input.strategy):
            if 'tabular_graphics_data' in line:
                if not self.tabular_graphics_data:
                    self.input.strategy[i] = \
                        line.replace('tabular_graphics_data', '')
                break
        else:
            if self.tabular_graphics_data:
                self.input.strategy.append('tabular_graphics_data')

        infile = self.get_pathname() + '.in'
        self.input.write_input(infile)
        try:
            run_dakota(infile, data=self, stdout=self.stdout, stderr=self.stderr)
        except Exception:
            self.reraise_exception()

    def dakota_callback(self, **kwargs):
        """
        Return responses from parameters.  `kwargs` contains:

        ========== ==============================================
        Key        Definition
        ========== ==============================================
        functions  number of functions (responses, constraints)
        ---------- ----------------------------------------------
        variables  total number of variables
        ---------- ----------------------------------------------
	cv         list/array of continuous variable values
        ---------- ----------------------------------------------
        div        list/array of discrete integer variable values
        ---------- ----------------------------------------------
        drv        list/array of discrete real variable values
        ---------- ----------------------------------------------
        av         single list/array of all variable values
        ---------- ----------------------------------------------
        cv_labels  continuous variable labels
        ---------- ----------------------------------------------
        div_labels discrete integer variable labels
        ---------- ----------------------------------------------
        drv_labels discrete real variable labels
        ---------- ----------------------------------------------
        av_labels  all variable labels
        ---------- ----------------------------------------------
        asv        active set vector (bit1=f, bit2=df, bit3=d^2f)
        ---------- ----------------------------------------------
        dvv        derivative variables vector
        ---------- ----------------------------------------------
        currEvalId current evaluation ID number
        ---------- ----------------------------------------------
        user_data  this object
        ========== ==============================================
        
        """
        cv = kwargs['cv']
        asv = kwargs['asv']
        self._logger.debug('cv %s', cv)
        self._logger.debug('asv %s', asv)

        self.set_parameters(cv)
        self.run_iteration()

        expressions = self.get_objectives().values()
        if hasattr(self, 'get_eq_constraints'):
            expressions.extend(self.get_eq_constraints().values())
        if hasattr(self, 'get_ineq_constraints'):
            expressions.extend(self.get_ineq_constraints().values())

        fns = []
        for i, expr in enumerate(expressions):
            if asv[i] & 1:
                result = expr.evaluate(self.parent)
                if isinstance(result, tuple):
                    lhs, rhs, op, violated = result
                    if '>' in op:
                        result = rhs - lhs
                    else:
                        result = lhs - rhs
                fns.append(result)
            if asv[i] & 2:
                self.raise_exception('Gradients not supported yet',
                                     NotImplementedError)
            if asv[i] & 4:
                self.raise_exception('Hessians not supported yet',
                                     NotImplementedError)

        retval = dict(fns=array(fns))
        self._logger.debug('returning %s', retval)
        return retval


@add_delegate(HasParameters, HasIneqConstraints, HasObjective)
class DakotaOptimizer(DriverUsesDerivatives, DakotaMixin):
    """ Optimizer using DAKOTA Python interface. """

    implements(IHasParameters, IHasIneqConstraints, IHasObjective, IOptimizer)

    max_iterations = Int(100, low=1, iotype='in')
    convergence_tolerance = Float(1.e-7, low=1.e-10, iotype='in')
    fd_gradient_step_size = Float(1.e-5, low=1.e-10, iotype='in')
    interval_type = Enum(values=('forward', 'central'), iotype='in')

    def __init__(self):
        DriverUsesDerivatives.__init__(self)
        DakotaMixin.__init__(self)

    def check_config(self):
        """ Verify valid configuration. """
        super(DakotaOptimizer, self).check_config()

        parameters = self.get_parameters()
        if not parameters:
            self.raise_exception('No parameters, run aborted', ValueError)

        objectives = self.get_objectives()
        if not objectives:
            self.raise_exception('No objectives, run aborted', ValueError)

    def execute(self):
        """ Write DAKOTA input and run. """
        ineq_constraints = self.get_ineq_constraints()
        objectives = self.get_objectives()
        method = 'conmin_mfd' if ineq_constraints else 'conmin_frcg'

        self.input.method = [
            '%s' % method,
            '    max_iterations = %s' % self.max_iterations,
            '    convergence_tolerance = %s' % self.convergence_tolerance,
        ]
        self.set_variables(need_start=True)
        self.input.responses = [
            'objective_functions = %s' % len(objectives)]
        if ineq_constraints:
            self.input.responses.append(
                'nonlinear_inequality_constraints = %s' % len(ineq_constraints))
        self.input.responses.extend([
            'numerical_gradients',
            '    method_source dakota',
            '    interval_type %s' % self.interval_type,
            '    fd_gradient_step_size = %s' % self.fd_gradient_step_size,
            '    no_hessians',
        ])
        self.run_dakota()


@add_delegate(HasParameters, HasObjectives)
class DakotaMultidimStudy(Driver, DakotaMixin):
    """ Multidimensional parameter study using DAKOTA Python interface. """

    implements(IHasParameters, IHasObjectives)

    partitions = List(Int, low=1, iotype='in')

    def __init__(self):
        Driver.__init__(self)
        DakotaMixin.__init__(self)

    def check_config(self):
        """ Verify valid configuration. """
        super(DakotaMultidimStudy, self).check_config()

        parameters = self.get_parameters()
        if not parameters:
            self.raise_exception('No parameters, run aborted', ValueError)

        if len(self.partitions) != len(parameters):
            self.raise_exception('#partitions (%s) != #parameters (%s)'
                                 % (len(self.partitions), len(parameters)),
                                 ValueError)

        objectives = self.get_objectives()
        if not objectives:
            self.raise_exception('No objectives, run aborted', ValueError)

    def execute(self):
        """ Write DAKOTA input and run. """
        partitions = [str(partition) for partition in self.partitions]
        objectives = self.get_objectives()

        self.input.method = [
            'multidim_parameter_study',
            '    partitions = %s' % ' '.join(partitions)
        ]
        self.set_variables(need_start=False)
        self.input.responses = [
            'objective_functions = %s' % len(objectives),
            'no_gradients',
            'no_hessians',
        ]
        self.run_dakota()


@add_delegate(HasParameters, HasObjectives)
class DakotaVectorStudy(Driver, DakotaMixin):
    """ Vector parameter study using DAKOTA Python interface. """

    implements(IHasParameters, IHasObjectives)

    final_point = List(Float, iotype='in')
    num_steps = Int(1, low=1, iotype='in')

    def __init__(self):
        Driver.__init__(self)
        DakotaMixin.__init__(self)

    def check_config(self):
        """ Verify valid configuration. """
        super(DakotaVectorStudy, self).check_config()

        parameters = self.get_parameters()
        if not parameters:
            self.raise_exception('No parameters, run aborted', ValueError)

        if len(self.final_point) != len(parameters):
            self.raise_exception('#final_point (%s) != #parameters (%s)'
                                 % (len(self.final_point), len(parameters)),
                                 ValueError)

        objectives = self.get_objectives()
        if not objectives:
            self.raise_exception('No objectives, run aborted', ValueError)

    def execute(self):
        """ Write DAKOTA input and run. """
        final_point = [str(point) for point in self.final_point]
        objectives = self.get_objectives()

        self.input.method = [
            'vector_parameter_study',
            '    final_point = %s' % ' '.join(final_point),
            '    num_steps = %s' % self.num_steps,
        ]
        self.set_variables(need_start=False)
        self.input.responses = [
            'objective_functions = %s' % len(objectives),
            'no_gradients',
            'no_hessians',
        ]
        self.run_dakota()

