
================
Package Metadata
================

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **description-file:** README.txt

- **entry_points**:: 

    [openmdao.component]
    dakota_driver.driver.DAKOTAoptimizer=dakota_driver.driver:DAKOTAoptimizer
    dakota_driver.driver.DAKOTAstudy=dakota_driver.driver:DAKOTAstudy
    [openmdao.driver]
    dakota_driver.driver.DAKOTAoptimizer=dakota_driver.driver:DAKOTAoptimizer
    dakota_driver.driver.DAKOTAstudy=dakota_driver.driver:DAKOTAstudy
    [openmdao.container]
    dakota_driver.driver.DAKOTAoptimizer=dakota_driver.driver:DAKOTAoptimizer
    dakota_driver.driver.DAKOTAstudy=dakota_driver.driver:DAKOTAstudy

- **home-page:** https://github.com/OpenMDAO-Plugins/dakota-driver

- **keywords:** openmdao

- **license:** Apache License, Version 2.0

- **name:** dakota_driver

- **requires-dist:** openmdao.main

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **summary:** 'OpenMDAO drivers using DAKOTA (Design Analysis Kit for Optimization and Terascale Applications)'

- **version:** 0.1

