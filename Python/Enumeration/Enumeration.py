from enum import Enum

# enumeration as a class
class enumeration(Enum):
    VALUE_1 = 0
    VALUE_2 = 1
    VALUE_3 = 2
    VALUE_4 = 3

# enumeration as a function
enumeration = Enum("enumeration", ["VALUE_1", "VALUE_2", "VALUE_3", "VALUE_4"])
    
if __name__ == "__main__":
    print("Python Enumeration Demo")
    
    value = enumeration.VALUE_1
    print(f"value: {value}")
    
    print(f"enumeration available values: {list(enumeration)}")