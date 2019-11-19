# Update Log

## Note

* Version identification format: `<main>.<sub>.<patch>(a|b|c|rc)<update>`.

## Pre-release 0.4: "Rocket"

* 2019-11-19 ![Version](https://img.shields.io/badge/pre--release-0.4.3a6-brightgreen.svg)
  * **BUG**: *Fixed*. Fixed not setting initial max distortion energy value.
  * **DEV**: *Improvement*. Eliminate unnecessary bond stretch calculation.
  * **DEV**: *New Feature*. Add unbroken zone.
  * **BUG**: *Fixed*. Fixed errorous adjoint node_id.
  * **DEV**: *Improvement*. Add `gauss_point_linear_standard`.
  * **DEV**: *New Feature*. Add load boundary condition treatments.
  * **DEV**: *Improvement*. Refactor code of boundary condition treatment.
  * **DEV**: *New Feature*. Add `onrange_in` and `onrange_ex`.
  * **BUG**: *Fixed*. Fixed the bond stretch code bug. The broken bond does not have stretch attribute.
* 2019-10-10 ![Version](https://img.shields.io/badge/pre--release-0.4.3a5-brightgreen.svg)
  * **DEV**: *Improvement*. Vectorization for PD stiffness matrix.
  * **DEV**: *New Feature*. Add `setOrthotropic` into `Material2d` and `PdMaterial2d` for orthotropic material.
* 2019-10-09 ![Version](https://img.shields.io/badge/pre--release-0.4.2a4-brightgreen.svg)
  * **DEV**: *Improvement*. Integrated output data.
  * **DEV**: *Improvement*. Improved method for calculating distortion energy.
  * **BUG**: *Fixed*. Improved calculation accuracy.
  * **DEV**: *New Feature*. Add module `algebra` to storage the related method.
* 2019-10-01 ![Version](https://img.shields.io/badge/pre--release-0.4.2a3-brightgreen.svg)
  * **DEV**: *New Feature*. Add computing geometry about line segments.
  * **DEV**: *Code Beautification*. Add a hint when the conversion of gmsh input into mesh is done.
  * **DEV**: *New Feature*. Add class `HybridCrackMesh2d` to handle the region that contains cracks.
* 2019-09-30 ![Version](https://img.shields.io/badge/pre--release-0.4.1a2-brightgreen.svg)
  * **DEV**: *New Feature*. It can now product local damage contour.
  * **DEV**: *New Feature*. Add `hmsolver.utils.SingletonDecorator`.
  * **DEV**: *Code Beautification*. Refactoring the PD Stiffness Matrix module.
  * **BUG**: *Fixed*. It will export the simulation solution file with same width of phase_id.
* 2019-09-27 ![Version](https://img.shields.io/badge/pre--release-0.4.0a1-brightgreen.svg)
  * **DOC**: *Improvement*. Add the mesh data of example-04.
  * **DEV**: *Improvement*. It's now using another way to run the simulation lazily. It runs until it needs.
  * **DEV**: *Code Beautification*. I rewrote the weighting function section to make it clearer than before.
* 2019-09-18 ![Version](https://img.shields.io/badge/pre--release-0.4.0a0-brightgreen.svg)
  * It's a stable version and much faster than before, so upgrade into 0.4*
  * **DEV**: *Improvement*. It now obeys [PEP400](https://www.python.org/dev/peps/pep-0440/) for version identification and dependency specification.
  * **DEV**: *Improvement*. It's now using another way to apply the boundary condition.
  * **DEV**: *Improvement*. It will return Stiffness Matrix and Loads Vector in stand of unsafe reference manipulating.
  * **DEV**: *Improvement*. It will use `scipy.sparse` and `scipy.sparse.linalg` to assemble Stiffness Matrix. It's much faster, saving about 30% time cost.
  * **DEV**: *Improvement*. Because of the defination of bonds, it can just run half of bond stretch test. This trick saves about 40% time cost.
  * **DEV**: *Improvement*. Move the useless code to recycle bin folder.

## Pre-release 0.3: "Sparse"

* 2019-09-12 ![Version](https://img.shields.io/badge/pre--release-0.3.1a90912-brightgreen.svg)
  * **BUG**: *Fixed*. It will no longer use the node which not belong to any element when use function hmsolver.femcore.preprocessing.convert_gmsh_into_msh.
  * **BUG**: *Fixed*. It will rerun the solution while the new one comes.
  * **DEV**: *Maybe improve in future*. It's now using `numpy.linalg.pinv` to solve linear system, by avoiding the singular matrix problem.
* 2019-07-30 ![Version](https://img.shields.io/badge/pre--release-0.3.0a90730-brightgreen.svg)
  * It's a stable version with tutorial, so upgrade into 0.3
  * **BUG**: *Fixed*. Markdown files is now able to provided to PyPI without url mistake

## Pre-release 0.2: "Refactor"

* 2019-07-30 ![Version](https://img.shields.io/badge/pre--release-0.2.2a90730-brightgreen.svg)
  * **DEV**: add hmsolver.app.simulation module
  * **DOC**: add handbook(in Simplified Chinese)
* 2019-07-27 ![Version](https://img.shields.io/badge/pre--release-0.2.1.a90727-brightgreen.svg)
  * Refactoring complete, upgrade into 0.2


## Prototype 0.1: "Scripts"

* 2019-07-14 ![Version](https://img.shields.io/badge/prototype-0.1-red.svg)
  * Upload the script version of Hybrid Model Solver
