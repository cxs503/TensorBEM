# OpenBEM

OpenBEM is an open-source framework for electromagnetic simulation
with the boundary element method.

OpenBEM is not a solver. Rather, it is a _framework_ of components
that can be fit together in highly customizable ways, to allow users
to implement their own application-specific solvers. It is intended as
a research tool to facilitate the testing and prototyping of new
ideas, formulations, and algorithms, without the researcher having to
re-implement aspects of the boundary element method that are not
pertinent to their specific research.

OpenBEM is written in C++, and its full documentation is available at
[https://shash-sharma.github.io/OpenBEM/](https://shash-sharma.github.io/OpenBEM/).

Source code is available on [GitHub](https://github.com/shash-sharma/OpenBEM).

As a starting point, it is highly recommended that users go through
the provided examples in the
[documentation](https://shash-sharma.github.io/OpenBEM/). To
gain familiarity with the code and its organization, the Topics page
in the
[documentation](https://shash-sharma.github.io/OpenBEM/)
is the recommended way to navigate through the available
functionality.

Citing OpenBEM: [![DOI](https://zenodo.org/badge/1060257425.svg)](https://doi.org/10.5281/zenodo.19337991)


# Features

### Features currently available

- All RWG-based operators associated with the TEFIE, NEFIE, TMFIE, and
  NMFIE [[Ylä-Oijala, Taskinen, Järvenpää, 2005]](https://ieeexplore.ieee.org/document/7770136).

- Numerical quadrature over source and observation triangles for
  RWG-based meshes and single and double layer kernels [[Ergül, Gürel, 2014]](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118844977).

- Singularity extraction for single and double layer kernels [[Ergül, Gürel, 2014]](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118844977).

- Line integration for the case of highly oscillatory kernels, such as
  those associated with lossy conductors [[Qian, Chew, Suaya, 2007]](https://ieeexplore.ieee.org/abstract/document/4359102) and [[Xia _et. al_, 2017]](https://ieeexplore.ieee.org/document/7955080).

- Plane wave, infinitesimal gap, and lumped circuit port excitations
  [[Gibson 2021]](https://www.taylorfrancis.com/books/mono/10.1201/9780429355509/method-moments-electromagnetics-walton-gibson), [[Wang _et. al_, 2004]](https://ieeexplore.ieee.org/document/1310664).

- Near- and far-field projection for post-processing.

- Input and output of [Gmsh](https://gmsh.info) meshes and fields for
  post processing.

- Sparse and dense matrix wrappers for the [Eigen library](https://libeigen.gitlab.io/eigen/docs-5.0.1/GettingStarted.html)
  for operator assembly.

- The ability to partition meshes and specify subsets of triangle
  pairs for which to compute integrals, to enable compatibility with
  mesh distribution and acceleration algorithms.

### Features coming "soon"

- A built-in standard acceleration algorithm (Q3-Q4 2026).

- Support for dyadic kernels (Q4 2026).

- Python bindings (Q1-Q2 2027).


# Design Philosophy

OpenBEM is designed to be modular and extensible, so that researchers
can focus on aspects of the boundary element method pertinent to their
research, without needing to implement other aspects that are not of
interest to them, but still must be implemented. The goal is to enable
researchers to prototype their ideas and plug them into the rest of
OpenBEM's functionality, and quickly assemble full electromagnetic
solvers to test their ideas.

For example,

- researchers focusing on high-level integral equation formulations
need not worry about having to implement integration techniques over
triangles for singular kernels;

- researchers developing new integration techniques for singular
kernels need not implement the surrounding code infrastructure needed
to test their ideas in the context of a full solver;

- researchers developing new acceleration algorithms that approximate
far-zone electromagnetic interactions can wrap OpenBEM's functionality
to compute the near-zone interactions.

This modularity is achieved with the following design pattern based on
the well-known concept of runtime polymorphism.

Each module contains a `base.hpp` file containing an abstract base
class for that module, for example,

```cpp
// module/base.hpp

class ModuleBase
{
// ...
};
```

The base classes contain a blueprint for the basic functionality that
any of its subclasses is expected to have. For example,

```cpp
// module/base.hpp

class ModuleBase
{
    virtual float compute_something(int input_1, float input_2) = 0;
// ...
};
```

This means that any class that inherits from `ModuleBase` must have a
function called `compute_something` with exactly the same return and
input arguments. For example, we can define a custom class that
implements our own idea of `compute_something`,

```cpp
// module/my_module.hpp

class MyModule: public ModuleBase
{
    float compute_something(int input_1, float input_2)
    {
        // implementation of my amazing research idea
    };
// ...
};
```

Since `MyModule` follows all the rules of `ModuleBase`, we can now use
`MyModule` throughout the rest of OpenBEM! For example, suppose
there's another OpenBEM function `func` which accepts a `ModuleBase`
reference as an input,

```cpp
// func.hpp

float func(ModuleBase& module)
{
    float output = module.compute_something(4, 5.0);
    return output;
}
```

then we can pass in our custom `MyModule` to this OpenBEM function,
for example,

```cpp
// test_mymodule.cpp

MyModule my_module;
float final_output = func(my_module);

// ...
```

The idea is that `func` doesn't care what you pass in as long as it
inherits from and follows the rules of `ModuleBase` (i.e., provides an
implementation of all methods that are marked virtual and set to 0).


# Requirements

OpenBEM requires at least C++17.

To compile your project with OpenBEM, the following should be noted:

- GCC version 10 or above, Clang version 8 or above, or an equivalent
  compiler is recommended.

- OpenBEM uses a relatively recent version of the [Eigen
  library](https://libeigen.gitlab.io/eigen/docs-5.0.1/GettingStarted.html).
  This is a header-only linear algebra toolkit which is included as
  part of the OpenBEM distribution, so you do not need to download or
  include it separately. However, if using an older compiler, you may
  need to disable certain compile options such as `march=native`,
  `mavx`, etc., for compatibility with this version of Eigen. I've
  experienced this issue with GCC 8.5.


# Usage

There are two ways to incorporate OpenBEM in your project.


## Option 1: Header-only mode

The simplest and quickest way to use OpenBEM is to import it into your
project it in header-only mode, which involves no installation,
building or linking; you simply `#include` the appropriate header into
your C++ files, and you're all set.

For example, if you want to include OpenBEM in a C++ file called
`main.cpp`, your compilation command may look like

```make
g++ -I/path/to/OpenBEM/source -std=c++17 main.cpp -o main
```

where the `-I` specifier adds the `source` directory in OpenBEM to the
compiler's include search path.

If you're using [CMake](https://cmake.org) to build your project
(recommended), you can include OpenBEM in header-only mode by doing
the following:

```CMake
add_executable(main main.cpp)
target_include_directories(main PRIVATE /path/to/OpenBEM/source)
target_compile_features(main PRIVATE cxx_std_17)
```

In either case, OpenBEM's headers can then be included
directly into your project, for example,

```cpp
#include "openbem.hpp"
```

and that's it.


## Option 2: Dynamically linked mode

The compile time of your project can be somewhat improved by using
OpenBEM in linked mode, which involves building and dynamically
linking OpenBEM to your project.

Doing this requires you to have at least version 3.16.0 of
[CMake](https://cmake.org) installed on your machine. Then, just run

```bash
cmake -B /path/to/build/directory -S /path/to/OpenBEM
make -C /path/to/build/directory
```

This will build OpenBEM in any `/path/to/build/directory` of your
choice.

The following options are available to use in the `cmake` command to
customize your build, by augmenting the `cmake` command as

```bash
cmake -DOPTION=VAL -B /path/to/build/directory -S /path/to/OpenBEM
```

where `OPTION` is one of the following options, and `VAL` is one of
the possible corresponding values for that option:

- `PRECISION`, which can be `single`, `double` (default), or `extended`.
- `DEBUG_LEVEL`, which can be `release` (default, optimized build) or `debug`.
- `BUILD_EXAMPLES`, which can be `yes` or `no` (default).
- `BUILD_TESTS`, which can be `yes` or `no` (default).

Once built, you still need to follow the steps in Option 1 to add the
include path to your project's compilation. In addition, you'll need
to link your project to the OpenBEM dynamic library
`/path/to/build/directory/libopenbem.[so/dylib/dll]`.

Then you can include OpenBEM's headers in your project as usual, for
example,

```cpp
#include "openbem.hpp"
```


# Citing

If OpenBEM benefits your research, please cite this GitHub repository via the following [![DOI](https://zenodo.org/badge/1060257425.svg)](https://doi.org/10.5281/zenodo.19337991) in your publications and presentations.