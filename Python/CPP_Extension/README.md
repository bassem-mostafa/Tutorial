# Build Python C++ extension (Windows)

---

## Pre-requisite(s)

1) Familiar with Python [C-API](https://docs.python.org/3/c-api/)

2) Familiar with Visual Studio Environment

---

## Step(s)

1) Create C/C++ source files

2) C/C++ source files MUST contain the following (which makes link to Python environment):

```cpp
	#include <Python.h>
	
	<...>
	
	static PyMethodDef <Method Table>[] = {
	    {"<Method Name In Python>", <Method Reference In C/C++>, <Method Flags>, "<Method Description>"},
		 <...>,
	    {NULL, NULL, 0, NULL}
	};
	
	static struct PyModuleDef <Module Structure> = {
	    PyModuleDef_HEAD_INIT,
	    "<Module Name>",
	    "<Module Description>",
	    -1,
		<Method Table Reference>
	};
	
	PyMODINIT_FUNC PyInit_<Extension Name>(void) {
	    return PyModule_Create(&<Module Structure Reference>);
	}
```

3) Create setup.py script with the following:

```python
from distutils.core import setup, Extension

def main():
    setup(name="<Module Name>",
          version="<Module Version>",
          description="<Module Description>",
          author="<Module Author>",
          author_email="<Module Author E-Mail>",
          ext_modules=[Extension("<Extension Name>", # <Extension Name> Must Match PyInit_<Extension Name> in your C/C++
          							["<C/C++ Source File>", "<...>"])])

if __name__ == "__main__":
    main()

```

4) Run setup.py either directly or using any package manager such as PIP 

```
cd <path to the directory contains `setup.py`>
pip install .
```

5) If reached so far without any issue(s), your module should be ready to be used in Python

---

# Example(s)
- [realpython](https://realpython.com/build-python-c-extension-module/#extending-your-python-program)

---

# Reference(s)
- [distutils](https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension)
- [setuptools](https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference)
- [visual studio](https://learn.microsoft.com/en-us/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2022)

---

# Fix(es)
- [eclipse visual c++ fix](https://stackoverflow.com/questions/41401515/visual-studio-toolchain-in-eclipse-for-c)