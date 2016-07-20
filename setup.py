from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('template_topology_feature', ['template_topology_feature.pyx'])]   #assign new.pyx module in setup.py.
setup(
     name        = 'template_topology_feature app',
     cmdclass    = {'build_ext':build_ext},
     ext_modules = ext_modules
     )

