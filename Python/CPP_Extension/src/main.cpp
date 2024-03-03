#include <Python.h>
#include <iostream>
using namespace std;

PyMODINIT_FUNC PyInit_MyModule(void);

int main() {
	cout << "\nInitializing Python Interpreter...";
	Py_Initialize();

	cout << (Py_IsInitialized() ? "\nInitialized": "\nFailed");

	cout << "\nInitializing MyModule...";
	PyObject* self = PyInit_MyModule();

	cout << "\nAvailable attributes\n\n";
	PyObject* elem = PyObject_Dir(self);
	PyObject_Print(elem,stdout,0);
	cout << endl;

	cout << "\nCalling API _hello\n\n";
	PyObject_CallMethod(self, "_hello", NULL);

	cout << "\nDe-Initializing Python Interpreter...";
	Py_Finalize();
	cout << (!Py_IsInitialized() ? "\nDe-Initialized": "\nFailed");
	return 0;
}
