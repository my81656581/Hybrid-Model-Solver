# Hybrid-Model-Solver

![Platfrom](https://img.shields.io/badge/python-3.5+-3572A5.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/pre--release-0.4.0a1-brightgreen.svg)
[![PyPI](https://img.shields.io/badge/PyPI-hmsolver-blue.svg)](https://pypi.org/project/hmsolver/)

## What is This

A Simple Simulation Tool by using morphing strategy to simulate the crack's development, static fracture, and so on.

## How to Use

* use PyPI to install/keep updated, type `pip install hmsolver` or `pip install hmsolver==<VERSION>` for specific version of it
* check [`example`](https://github.com/polossk/Hybrid-Model-Solver/tree/master/example) folder first to know how to use this tool
* I have provided a short version of manual in Chinese, go and check [this](https://github.com/polossk/Hybrid-Model-Solver/tree/master/Chinese-Handbook) First

## Update Log(Most Recent)

![Version](https://img.shields.io/badge/pre--release-0.4.0a1-brightgreen.svg)
* **DOC**: *Improvement*. Add the mesh data of example-04.
* **DEV**: *Improvement*. It's now using another way to run the simulation lazily. It runs until it needs.
* **DEV**: *Code Beautification*. I rewrote the weighting function section to make it clearer than before.

[(read more)](https://github.com/polossk/Hybrid-Model-Solver/tree/master/update-log.md)

## Copyright

Use this code whatever you want, under the circumstances of acknowleged the
mit license this page below. Star this repository if you like, and it will
be very generous of you!

## Reference

* Azdoud, Y., Han, F., & Lubineau, G. (2014). The morphing method as a flexible tool for adaptive local/non-local simulation of static fracture. *Computational Mechanics*, 54(3), 711-722. doi [10.1007/s00466-014-1023-3](https://doi.org/10.1007/s00466-014-1023-3)
* Azdoud, Y., Han, F., & Lubineau, G. (2013). A Morphing framework to couple non-local and local anisotropic continua. *International Journal of Solids and Structures*, 50(9), 1332-1341. doi [10.1016/j.ijsolstr.2013.01.016](https://doi.org/10.1016/j.ijsolstr.2013.01.016)
* Lubineau, G., Azdoud, Y., Han, F., Rey, C., & Askari, A. (2012). A morphing strategy to couple non-local to local continuum mechanics. *Journal of The Mechanics and Physics of Solids*, 60(6), 1088-1102. doi [10.1016/j.jmps.2012.02.009](https://doi.org/10.1016/j.jmps.2012.02.009)

## License

The MIT License (MIT)

Copyright (c) 2019 Shangkun Shen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
