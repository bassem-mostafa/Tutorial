#include <Python.h>
#include <iostream>
using namespace std;

static PyObject* _hello_from_cpp( PyObject *self, PyObject *Py_UNUSED(args) ) {
	cout << "C++ power is now integrated with Python" << endl;
	Py_RETURN_NONE;
}

static PyMethodDef Module_Methods[] = {
    {"_hello", (PyCFunction)_hello_from_cpp, METH_NOARGS, "prints a message using C++ iostream cout std library"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "Module",
    "Python extension for C++",
    -1,
	Module_Methods
};

PyMODINIT_FUNC PyInit_MyModule(void) {
    return PyModule_Create(&Module);
}
