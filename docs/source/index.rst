.. regressioninc documentation master file, created by
   sphinx-quickstart on Wed Oct 19 21:49:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RegressionInC docs
==================

*Regression*\ In\ *C* (regressioninc) is a package focussing on regression for
complex-valued data.

There are already multiple packages for regression in Python, however these
offer varying amounts of support for the complex-valued variables. Scipy's least
squares implementation has partial support for complex-valued variables and
other packages support multi-target regression, which can be used for
complex-valued problems, but again support is spotty. |pkgnm| fills the gap for
complex-valued regression.

.. warning::

   This package is currently being developed and its API is liable to change in
   the early stages.

Installation
------------

|pkgnm| can be installed from
`PYPI <https://pypi.org/project/regressioninc/>`_ as normal.

.. code::

   pip install regressioninc

Or if you are a poetry user

.. code::

   poetry add regressioninc

Next steps
----------

First time users and those new to complex-valued regression are directed towards
the introduction page and basic exmaples.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   introduction.rst
   terminology.rst
   examples_complex_regression/index.rst
   examples_linear_models/index.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   regressioninc.rst

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   development.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
