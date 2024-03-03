# generator as a class
class generator:
    def __init__(self):
        self.i = 10
    def __next__(self):
        result = self.i
        self.i += 1
        return result
    def __iter__(self):
        return self

## generator as a function
# def generator():
#     i = 10
#     while True:
#         result = i
#         i += 1
#         yield result
    
if __name__ == "__main__":
    print("Python Generator Demo")
    for i, elem in zip(range(10), generator()):
        print(f"[{i}]: {elem}")