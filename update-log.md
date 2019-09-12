# Update Log

* 2019-09-12 ![Version](https://img.shields.io/badge/pre--release-0.3.1.a90912-brightgreen.svg)
  * **BUG**: *Fixed*. It will no longer use the node which not belong to any element when use function hmsolver.femcore.preprocessing.convert_gmsh_into_msh.
  * **BUG**: *Fixed*. It will rerun the solution while the new one comes.
  * **DEV**: *Maybe impove in future*. It's now using numpy.linalg.pinv to solve linear system, by avoiding the singular matrix problem.
* 2019-07-30 ![Version](https://img.shields.io/badge/pre--release-0.3.0.a90730-brightgreen.svg)
  * It's a stable version with tutorial, so upgrade into 0.3
  * **BUG**: *Fixed*. Markdown files is now able to provided to PyPI without url mistake
* 2019-07-30 ![Version](https://img.shields.io/badge/pre--release-0.2.2.a90730-brightgreen.svg)
  * **DEV**: add hmsolver.app.simulation module
  * **DOC**: add handbook(in Simplified Chinese)
* 2019-07-27 ![Version](https://img.shields.io/badge/pre--release-0.2.1.a90727-brightgreen.svg)
  * Refactoring complete, upgrade into 0.2
* 2019-07-14 ![Version](https://img.shields.io/badge/prototype-0.1-red.svg)
  * Upload the script version of Hybrid Model Solver
