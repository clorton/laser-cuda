# README - PyCUDA Test

- Run `code .` in the piecoodah directory after running `"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"` in order to have access to the VS2022 compiler.

- function parameters should be NumPy types
- block and grid parameters should by Python int(egers)
- mixed parameters, 32-bit and 64-bit, don't play well
- int64/uint64 and doubles work better (Python struct packing?)
- https://stackoverflow.com/questions/6954487/how-to-use-the-prepare-function-from-pycuda
