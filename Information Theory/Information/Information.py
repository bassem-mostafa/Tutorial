from math import log2

def information(event):
    return -log2(event)

if __name__ == '__main__':
    print('Information Demo')
    
    prob = [1/10, 9/10] # MUST sum to ONE
    prob = list(map(lambda p: p / sum(prob), prob)) # make sure prob sums to ONE
    for i, p in enumerate(prob):
        print(f'p(c{i}): {p}, have information: {information(p): 6.3f}')