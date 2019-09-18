from setuptools import setup

with open("README.md", 'r', encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='hmsolver',
      version='0.4.0a0',
      description='Hybrid Model Solver',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/polossk/Hybrid-Model-Solver',
      author='Shangkun Shen(polossk)',
      author_email='poloshensk@gmail.com',
      license='MIT License',
      packages=[
          'hmsolver', 'hmsolver.app', 'hmsolver.basis', 'hmsolver.femcore',
          'hmsolver.material', 'hmsolver.meshgrid'
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ])
