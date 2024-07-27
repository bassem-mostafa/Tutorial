from distutils.core import setup, Extension

def main():
    setup(name="ExtensionModule",
          version="1.0.0",
          description="Python C/C++ extension",
          author="BaSSeM",
          author_email=None,
          ext_modules=[Extension("MyModule",        # MUST match PyInit_<MyModule>
                                 ["src/Module.cpp"])])

if __name__ == "__main__":
    main()
