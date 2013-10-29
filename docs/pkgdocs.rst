
================
Package Metadata
================

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **description-file:** README.txt

- **entry_points**:: 

    [openmdao.component]
    dakota_driver.driver.DakotaOptimizer=dakota_driver.driver:DakotaOptimizer
    test_driver.Textbook=test_driver:Textbook
    dakota_driver.driver.DakotaVectorStudy=dakota_driver.driver:DakotaVectorStudy
    test_driver.Broken=test_driver:Broken
    test_driver.SensitivityStudy=test_driver:SensitivityStudy
    test_driver.Rosenbrock=test_driver:Rosenbrock
    test_driver.ConstrainedOptimization=test_driver:ConstrainedOptimization
    dakota_driver.driver.DakotaGlobalSAStudy=dakota_driver.driver:DakotaGlobalSAStudy
    dakota_driver.driver.DakotaBase=dakota_driver.driver:DakotaBase
    test_driver.ParameterStudy=test_driver:ParameterStudy
    test_driver.VectorStudy=test_driver:VectorStudy
    test_driver.Optimization=test_driver:Optimization
    dakota_driver.driver.DakotaMultidimStudy=dakota_driver.driver:DakotaMultidimStudy
    [openmdao.driver]
    dakota_driver.driver.DakotaOptimizer=dakota_driver.driver:DakotaOptimizer
    dakota_driver.driver.DakotaVectorStudy=dakota_driver.driver:DakotaVectorStudy
    dakota_driver.driver.DakotaBase=dakota_driver.driver:DakotaBase
    dakota_driver.driver.DakotaGlobalSAStudy=dakota_driver.driver:DakotaGlobalSAStudy
    dakota_driver.driver.DakotaMultidimStudy=dakota_driver.driver:DakotaMultidimStudy
    [openmdao.container]
    test_driver.Textbook=test_driver:Textbook
    dakota_driver.driver.DakotaOptimizer=dakota_driver.driver:DakotaOptimizer
    dakota_driver.driver.DakotaVectorStudy=dakota_driver.driver:DakotaVectorStudy
    test_driver.Broken=test_driver:Broken
    test_driver.SensitivityStudy=test_driver:SensitivityStudy
    test_driver.Rosenbrock=test_driver:Rosenbrock
    test_driver.ConstrainedOptimization=test_driver:ConstrainedOptimization
    dakota_driver.driver.DakotaGlobalSAStudy=dakota_driver.driver:DakotaGlobalSAStudy
    dakota_driver.driver.DakotaBase=dakota_driver.driver:DakotaBase
    test_driver.ParameterStudy=test_driver:ParameterStudy
    test_driver.VectorStudy=test_driver:VectorStudy
    test_driver.Optimization=test_driver:Optimization
    dakota_driver.driver.DakotaMultidimStudy=dakota_driver.driver:DakotaMultidimStudy

- **home-page:** https://github.com/OpenMDAO-Plugins/dakota_driver

- **keywords:** openmdao

- **license:** Apache License, Version 2.0

- **name:** dakota_driver

- **requires-dist**:: 

    openmdao.main
    pyDAKOTA

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **summary:** 'OpenMDAO drivers using DAKOTA (Design Analysis Kit for Optimization and Terascale Applications)'

- **version:** 0.2

