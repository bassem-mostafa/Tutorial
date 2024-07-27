from math import log2

if __name__ == '__main__':
    print('Information Cross Entropy Demo')
    
    # calculate cross entropy
    def cross_entropy(p, q):
        return -sum([p[i]*log2(q[i]) for i in range(len(p))])
     
    # define data
    p = [0.10, 0.40, 0.50]
    q = [0.80, 0.15, 0.05]
    # calculate cross entropy H(P, Q)
    ce_pq = cross_entropy(p, q)
    print('H(P, Q): %.3f bits' % ce_pq)
    # calculate cross entropy H(Q, P)
    ce_qp = cross_entropy(q, p)
    print('H(Q, P): %.3f bits' % ce_qp)
    # calculate cross entropy H(P, P)
    ce_pp = cross_entropy(p, p)
    print('H(P, P): %.3f bits' % ce_pp)
    # calculate cross entropy H(Q, Q)
    ce_qq = cross_entropy(q, q)
    print('H(Q, Q): %.3f bits' % ce_qq)