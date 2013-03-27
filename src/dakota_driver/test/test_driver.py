""" Test DAKOTA-based drivers. """

import nose
import os.path
import sys
import unittest

from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float
from openmdao.util.testutil import assert_rel_error

from dakota_driver import DAKOTAoptimizer, DAKOTAstudy


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


class Broken(Component):
    """ Always raises an exception. """

    x1 = Float(iotype='in')
    x2 = Float(iotype='in')
    f  = Float(iotype='out')

    def execute(self):
        """ Just raise a RuntimeError. """
        raise RuntimeError('Evaluating x1=%s, x2=%s' % (self.x1, self.x2))


class Optimization(Assembly):
    """ Use DAKOTA to perform optimization of Rosenbrock function. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock()) 

        driver = self.add('driver', DAKOTAoptimizer())
        driver.workflow.add('rosenbrock')

        driver.add_parameter('rosenbrock.x1', low=-2, high=2, start=-1.2)
        driver.add_parameter('rosenbrock.x2', low=-2, high=2, start=1)
        driver.add_objective('rosenbrock.f')


class ParameterStudy(Assembly):
    """ Use DAKOTA to perform parameter study on Rosenbrock function. """

    def configure(self):
        """ Configure driver and its workflow. """
        super(Assembly, self).configure()
        self.add('rosenbrock', Rosenbrock()) 

        driver = self.add('driver', DAKOTAstudy())
        driver.partitions = [5, 5]
        driver.workflow.add('rosenbrock')

        driver.add_parameter('rosenbrock.x1', low=-2, high=2)
        driver.add_parameter('rosenbrock.x2', low=-2, high=2)
        driver.add_objective('rosenbrock.f')


class TestCase(unittest.TestCase):
    """ Test DAKOTA-based drivers. """

    def tearDown(self):
        """ Cleanup files. """
        for name in ('dakota.rst', 'dakota_tabular.dat', 'driver.in'):
            if os.path.exists(name):
                os.remove(name)

    def test_optimization(self):
        # Test DAKOTAoptimizer driver.
        top = Optimization()
        top.run()
        # Last point isn't actually the optimium,
        # it's probably left over from CONMIN's line search.
        assert_rel_error(self, top.rosenbrock.x1, 0.99562777,     0.00001)
        assert_rel_error(self, top.rosenbrock.x2, 0.9911804,      0.00001)
        assert_rel_error(self, top.rosenbrock.f,  2.00049375e-05, 0.00001)

    def test_broken_optimization(self):
        # Test exception handling.
        # Disabled since this will terminate Python.
        raise nose.SkipTest('Terminates Python')
        top = Optimization()
        top.rosenbrock = Broken()
        top.run()

    def test_study(self):
        # Test DAKOTAstudy driver.
        top = ParameterStudy()
        top.run()
        self.assertEqual(top.rosenbrock.x1, 2)
        self.assertEqual(top.rosenbrock.x2, 2)
        self.assertEqual(top.rosenbrock.f,  401)

        top.driver.partitions = [8, 8]
        top.run()
        self.assertEqual(top.rosenbrock.x1, 2)
        self.assertEqual(top.rosenbrock.x2, 2)
        self.assertEqual(top.rosenbrock.f,  401)


if __name__ == '__main__':
    sys.argv.append('--cover-package=dakota_driver')
    sys.argv.append('--cover-erase')
    nose.runmodule()

