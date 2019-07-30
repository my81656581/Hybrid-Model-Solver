from setuptools import setup

setup(name='hmsolver',
      version='0.2.2.a90730',
      description='Hybrid Model Solver',
      url='https://github.com/polossk/Hybrid-Model-Solver',
      author='Shangkun Shen(polossk)',
      author_email='poloshensk@gmail.com',
      license='MIT License',
      packages=[
          'hmsolver', 'hmsolver.app', 'hmsolver.basis', 'hmsolver.femcore',
          'hmsolver.material', 'hmsolver.meshgrid'
      ])
