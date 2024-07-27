## #############################################################################
## #### Copyright ##############################################################
## #############################################################################

'''
Copyright (C) 2024 BaSSeM

This software is distributed under the terms and conditions of the 'Apache-2.0'
license which can be found in the file 'LICENSE.txt' in this package distribution
or at 'http://www.apache.org/licenses/LICENSE-2.0'.
'''

## #############################################################################
## #### Description ############################################################
## #############################################################################

## #############################################################################
## #### Control Variable(s) ####################################################
## #############################################################################

## #############################################################################
## #### Import(s) ##############################################################
## #############################################################################

## #############################################################################
## #### Private Type(s) ########################################################
## #############################################################################

class C1:
    def __init__(self, x):
        self._x = x

    def getx(self):
        return self._x

    def setx(self, value):
        raise AttributeError(f"{self.__class__.__name__}.x attribute access denied") # uncomment to disable attribute modification
        self._x = value

    def delx(self):
        del self._x

    # Here `x` is just a wrapper for the actual `_x` attribute
    # Also, `x` here IS A class property NOT instance property
    x = property(getx, setx, delx, "I'm the 'x' property.")

# C2 is equivalent to C1, just using python decorator syntactic sugar
class C2:
    def __init__(self, x):
        self._x = x
        
    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        raise AttributeError(f"{self.__class__.__name__}.x attribute access denied") # uncomment to disable attribute modification
        self._x = value

    @x.deleter
    def x(self):
        del self._x

# C3 is equivalent to C1 & C2, just using python builtin `__getattr__` and `__setattr__`
class C3:
    def __init__(self, x):
        self._x = x
    
    def __getattr__(self, name):
        # this API is more generic than the decorator, can control getting access to multiple attributes not just one
        # Also, this does NOT add class attribute/property, just manage/control the access behavior
        if name in ["x"]:
            return self._x
        return super().__getattr__(name)
    
    def __setattr__(self, name, value):
        # this API is more generic than the decorator, can control setting access to multiple attributes not just one
        # Also, this does NOT add class attribute/property, just manage/control the access behavior
        if name in ["x"]:
            raise AttributeError(f"{self.__class__.__name__}.x attribute access denied") # uncomment to disable attribute modification
            self._x = value
        return super().__setattr__(name, value)

## #############################################################################
## #### Private Method(s) Prototype ############################################
## #############################################################################

## #############################################################################
## #### Private Variable(s) ####################################################
## #############################################################################

## #############################################################################
## #### Private Method(s) ######################################################
## #############################################################################

## #############################################################################
## #### Public Method(s) Prototype #############################################
## #############################################################################

## #############################################################################
## #### Public Type(s) #########################################################
## #############################################################################

## #############################################################################
## #### Public Method(s) #######################################################
## #############################################################################

## #############################################################################
## #### Public Variable(s) #####################################################
## #############################################################################

## #############################################################################
## #### Main ###################################################################
## #############################################################################

if __name__ == "__main__":
    c1 = C1(10)
    c2 = C2(20)
    c3 = C3(30)
    
    print(f"Before Update")
    print(c1.x)
    print(c2.x)
    print(c3.x)

    # c1.x = 110 # This will raise an attribute error, if uncommented the raise expression
    # c2.x = 220 # This will raise an attribute error, if uncommented the raise expression
    # c3.x = 330 # This will raise an attribute error, if uncommented the raise expression
    
    print(f"After Update")
    print(c1.x)
    print(c2.x)
    print(c3.x)

## #############################################################################
## #### END OF FILE ############################################################
## #############################################################################