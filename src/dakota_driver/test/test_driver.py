""" Test DAKOTA-based drivers. """

import csv
import logging
import nose
import os.path
import sys
import unittest

from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float
from openmdao.util.testutil import assert_rel_error, assert_raises

from dakota_driver import DakotaOptimizer, DakotaMultidimStudy, \
                          DakotaVectorStudy


class Rosenbrock(Component):
    """ Standard two-dimensional Rosenbrock function. """

    x1 = Float(iotype='in')
    x2 = Float(iotype='in')
    f  = Float(iotype='out')

    def execute(self):
        """ Just evaluate the function. """
        x1 = self.x1
        x2 = self.x2
        self.f = 100 * (x2 - x1**2)**2 + (1 - x1)**2


class Textbook(Component):
    """ DAKOTA 'text_book' function. """

    x1 = Float(iotype='in')
    x2 = Float(iotype='in')
    f  = Float(iotype='out')

    def execute(self):
        """ Just evaluate the function. """
        self.f = (self.x1 - 1)**4 + (self.x2 - 1)**4


class Broken(Component):
    """ Always raises an exception. """

    x1 = Float(iotype='in')
    x2 = Float(iotype='in')
    f  = Float(iotype='out')

    def execute(self):
        """ Just raise a RuntimeError. """
        raise RuntimeError('Evaluating x1=%s, x2=%s' % (self.x1, self.x2))


class Optimization(Assembly):
    """ Use DAKOTA to perform an optimization. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock()) 

        driver = self.add('driver', DakotaOptimizer())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.tabular_graphics_data = True
        driver.max_iterations = 100
        driver.convergence_tolerance = 1e-4
        driver.interval_type = 'forward'
        driver.fd_gradient_step_size = 1e-5

        driver.add_parameter('rosenbrock.x1', low=-2, high=2, start=-1.2)
        driver.add_parameter('rosenbrock.x2', low=-2, high=2, start=1)
        driver.add_objective('rosenbrock.f')


class ConstrainedOptimization(Assembly):
    """ Use DAKOTA to perform a constrained optimization. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('textbook', Textbook()) 

        driver = self.add('driver', DakotaOptimizer())
        driver.workflow.add('textbook')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.max_iterations = 50
        driver.convergence_tolerance = 1e-4
        driver.interval_type = 'central'
        driver.fd_gradient_step_size = 1e-4

        driver.add_parameter('textbook.x1', low=0.5, high=5.8, start=0.9)
        driver.add_parameter('textbook.x2', low=-2.9, high=2.9, start=1.1)
        driver.add_objective('textbook.f')
        driver.add_constraint('textbook.x1**2 - textbook.x2/2 <= 0', name='g1')
        driver.add_constraint('textbook.x2**2 - textbook.x1/2 <= 0', name='g2')


class ParameterStudy(Assembly):
    """ Use DAKOTA to run a multidimensional parameter study. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock()) 

        driver = self.add('driver', DakotaMultidimStudy())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.partitions = [8, 8]

        driver.add_parameter('rosenbrock.x1', low=-2, high=2)
        driver.add_parameter('rosenbrock.x2', low=-2, high=2)
        driver.add_objective('rosenbrock.f')


class VectorStudy(Assembly):
    """ Use DAKOTA to run a vector study. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock()) 

        driver = self.add('driver', DakotaVectorStudy())
        driver.workflow.add('rosenbrock')
        driver.stdout = 'dakota.out'
        driver.stderr = 'dakota.err'
        driver.final_point = [1.1, 1.3]
        driver.num_steps = 10

        # Can't have parameter with no 'low' or 'high' defined.
        driver.add_parameter('rosenbrock.x1', start=-0.3, low=-1e10, high=1.e10)
        driver.add_parameter('rosenbrock.x2', start=0.2, low=-1e10, high=1.e10)
        driver.add_objective('rosenbrock.f')


class TestCase(unittest.TestCase):
    """ Test DAKOTA-based drivers. """

    def tearDown(self):
        """ Cleanup files. """
        for name in ('dakota.out', 'dakota.err',
                     'dakota.rst', 'dakota_tabular.dat', 'driver.in'):
            if os.path.exists(name):
                try:
                    os.remove(name)
                except WindowsError as exc:
                    # Currently no way to release DAKOTA streams.
                    logging.debug("Can't remove %s: %s", name, exc)

    def test_optimization(self):
        # Test DakotaOptimizer driver.
        logging.debug('')
        logging.debug('test_optimization')

        top = Optimization()
        top.run()
        # Current state isn't optimium,
        # probably left over from CONMIN line search.
        assert_rel_error(self, top.rosenbrock.x1, 0.99401209, 0.00001)
        assert_rel_error(self, top.rosenbrock.x2, 0.98869321, 0.00001)
        assert_rel_error(self, top.rosenbrock.f,  7.59464541e-05, 0.00001)

        with open('dakota_tabular.dat', 'rb') as inp:
            reader = csv.reader(inp, delimiter=' ', skipinitialspace=True)
            count = 0
            for row in reader:
                print row
                count += 1

        self.assertEqual(count, 83)
        self.assertEqual(row[0], '82')
        assert_rel_error(self, float(row[1]), 0.99401209, 0.00001)
        assert_rel_error(self, float(row[2]), 0.98869321, 0.00001)
        assert_rel_error(self, float(row[3]), 7.59464541e-05, 0.00001)

    def test_constrained_optimization(self):
        # Test DakotaOptimizer driver.
        logging.debug('')
        logging.debug('test_constrained_optimization')

        top = ConstrainedOptimization()
        top.run()
        # Current state isn't optimium,
        # probably left over from CONMIN line search.
        assert_rel_error(self, top.textbook.x1, 0.5, 0.0004)
        assert_rel_error(self, top.textbook.x2, 0.43167254, 0.0004)
        assert_rel_error(self, top.textbook.f,  0.16682649, 0.0007)

    def test_broken_optimization(self):
        # Test exception handling. This requires a modified version of
        # DAKOTA that can be configured to not exit on analysis failure.
        logging.debug('')
        logging.debug('test_broken_optimization')

#        raise nose.SkipTest('Requires abort_returns() modification to DAKOTA')
        top = Optimization()
        # Avoid messing-up file for test_optimization.
        top.driver.tabular_graphics_data = False
        top.rosenbrock = Broken()
        assert_raises(self, 'top.run()', globals(), locals(), RuntimeError,
                      'driver: Evaluating x1=-1.2, x2=1.0')

    def test_multidim(self):
        # Test DakotaMultidimStudy driver.
        logging.debug('')
        logging.debug('test_multidim')

        top = ParameterStudy()
        top.run()
        self.assertEqual(top.rosenbrock.x1, 2)
        self.assertEqual(top.rosenbrock.x2, 2)
        self.assertEqual(top.rosenbrock.f,  401)

    def test_vector(self):
        # Test DakotaVectorStudy driver.
        logging.debug('')
        logging.debug('test_vector')

        top = VectorStudy()
        top.run()
        assert_rel_error(self, top.rosenbrock.x1, 1.1, 0.00001)
        assert_rel_error(self, top.rosenbrock.x2, 1.3, 0.00001)
        assert_rel_error(self, top.rosenbrock.f,  0.82, 0.00001)


if __name__ == '__main__':
    sys.argv.append('--cover-package=dakota_driver')
    sys.argv.append('--cover-erase')
    nose.runmodule()

