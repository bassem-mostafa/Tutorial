from math import log2

def entropy(prob):
    '''
    The intuition for entropy is that it is 
    the average number of bits required to represent 
    or transmit an event drawn from the probability distribution 
    for the random variable.
    
    ex: for two variables c0, c1 with prob 0.5, 0.5 
    it leads to MAX entropy of 1.0 which means we need at least 1 bit
    
    ex: for two variables c0, c1 with prob 1.0, 0.0 
    it leads to MIN entropy of 0.0 which means we need at least 0 bit
    '''
    _epsilon = 1e-15
    prob = list(map(lambda p: p if p - _epsilon > 0 else _epsilon, prob)) # to avoid compute log(zero)
    return -sum( p * log2(p + _epsilon) for p in prob )

if __name__ == '__main__':
    print('Information Entropy Demo')
    
    prob = [1/10, 9/10] # MUST sum to ONE
    prob = list(map(lambda p: p / sum(prob), prob)) # make sure prob sums to ONE
    for i, p in enumerate(prob):
        print(f'p(c{i}): {p}')
    print(f'entropy: {entropy(prob): 6.3f}')