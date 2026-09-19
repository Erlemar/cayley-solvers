"""Test whether one circular cut always admits at most floor(n^2/4) inversions."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def enumerate_cuts(n):
    s=np.arange(n)
    worst=-1
    witness=s.copy()
    while True:
        best=n*n
        for t in range(n):
            inv=0
            for i in range(n):
                a=(s[(i+t)%n]-t)%n
                for j in range(i+1,n):
                    b=(s[(j+t)%n]-t)%n
                    inv+=a>b
            best=min(best,inv)
        if best>worst:
            worst=best
            witness=s.copy()
        k=n-2
        while k>=0 and s[k]>=s[k+1]: k-=1
        if k<0: break
        j=n-1
        while s[j]<=s[k]: j-=1
        s[k],s[j]=s[j],s[k]
        a,b=k+1,n-1
        while a<b:
            s[a],s[b]=s[b],s[a]
            a+=1;b-=1
    return worst,witness

if __name__=='__main__':
    for n in range(5,10):
        w,p=enumerate_cuts(n)
        print(n,w,n*n//4,p.tolist(),flush=True)
