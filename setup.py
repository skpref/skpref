try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='skpref',
      version=0.0.1,
      description='A machine learning toolbox focused on preference learning',
      author='The skpref team',
      packages=['skpref'],
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      )
