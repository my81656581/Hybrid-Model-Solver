# Update Log

* 2019-09-18 ![Version](https://img.shields.io/badge/pre--release-0.4.0a0-brightgreen.svg)
  * It's a stable version and much faster than before, so upgrade into 0.4*
  * **DEV**: *Improvement*. It now obeys [PEP400](https://www.python.org/dev/peps/pep-0440/) for version identification and dependency specification.
  * **DEV**: *Improvement*. It's now using another way to apply the boundary condition.
  * **DEV**: *Improvement*. It will return Stiffness Matrix and Loads Vector in stand of unsafe reference manipulating.
  * **DEV**: *Improvement*. It will use `scipy.sparse` and `scipy.sparse.linalg` to assemble Stiffness Matrix. It's much faster, saving about 30% time cost.
  * **DEV**: *Improvement*. Because of the defination of bonds, it can just run half of bond stretch test. This trick saves about 40% time cost.
  * **DEV**: *Improvement*. Move the useless code to recycle bin folder.
* 2019-09-12 ![Version](https://img.shields.io/badge/pre--release-0.3.1a90912-brightgreen.svg)
  * **BUG**: *Fixed*. It will no longer use the node which not belong to any element when use function hmsolver.femcore.preprocessing.convert_gmsh_into_msh.
  * **BUG**: *Fixed*. It will rerun the solution while the new one comes.
  * **DEV**: *Maybe improve in future*. It's now using `numpy.linalg.pinv` to solve linear system, by avoiding the singular matrix problem.
* 2019-07-30 ![Version](https://img.shields.io/badge/pre--release-0.3.0a90730-brightgreen.svg)
  * It's a stable version with tutorial, so upgrade into 0.3
  * **BUG**: *Fixed*. Markdown files is now able to provided to PyPI without url mistake
* 2019-07-30 ![Version](https://img.shields.io/badge/pre--release-0.2.2a90730-brightgreen.svg)
  * **DEV**: add hmsolver.app.simulation module
  * **DOC**: add handbook(in Simplified Chinese)
* 2019-07-27 ![Version](https://img.shields.io/badge/pre--release-0.2.1.a90727-brightgreen.svg)
  * Refactoring complete, upgrade into 0.2
* 2019-07-14 ![Version](https://img.shields.io/badge/prototype-0.1-red.svg)
  * Upload the script version of Hybrid Model Solver
