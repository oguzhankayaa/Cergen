# -*- coding: utf-8 -*-
"""

implementation of my own tensor library, called CerGen (short for CENG Gergen -- gergen: one of the Turkish translations of the term tensor).

Example usage:

```python
from cergen import rastgele_gercek,rastgele_dogal,cekirdek, gergen

boyut = ()
aralik = (0, 10)
g0 = rastgele_gercek(boyut, aralik)
print(g0)
0 boyutlu skaler gergen:
8

g1 = gergen([[1, 2, 3], [4, 5, 6]])
print(g1)
2x3 boyutlu gergen:
[[1, 2, 3]
 [4, 5, 6]]

g2 = gergen(rastgele_dogal((3, 1)))
print(g2)
3x1 boyutlu gergen
[[6],
[5],
[2]]

print((g1 * g2))


g3 = (g1 * (g2 + 3)).topla()

```

## 1 Task Description
we introduce the gergen class, a custom data structure designed to provide a
hands-on experience with fundamental array operations, mirroring some functionalities typically
found in libraries like NumPy.

## Fundamental Operations:
Random number generation:
"""

import random
def cekirdek(sayi: int):
    random.seed(sayi)

def boyut_acan_dogal(boyut,alt,ust):
  if len(boyut)==1:
    t=[]
    for i in range(boyut[0]):
      t.append(random.randint(alt,ust))
    return t
  else:
    t=[]
    for i in range(boyut[0]):
      t.append(boyut_acan_dogal(boyut[1:],alt,ust))
    return t

def boyut_acan_gercek(boyut,alt,ust):
  if len(boyut)==1:
    t=[]
    for i in range(boyut[0]):
      t.append(random.uniform(alt,ust))
    return t
  else:
    t=[]
    for i in range(boyut[0]):
      t.append(boyut_acan_gercek(boyut[1:],alt,ust))
    return t


def rastgele_dogal(boyut, aralik=(0,100), dagilim='uniform'):
  if dagilim!='uniform' :
    raise ValueError('dagilim misvalue')
  return gergen(boyut_acan_dogal(boyut,aralik[0],aralik[1]))


def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
  if dagilim!='uniform' :
    raise ValueError('dagilim misvalue')
  return gergen(boyut_acan_gercek(boyut,aralik[0],aralik[1]))

"""Operation class implementation:"""

class Operation:
    def __call__(self, *operands):
        """
        Makes an instance of the Operation class callable.
        Stores operands and initializes outputs to None.
        Invokes the forward pass of the operation with given operands.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the forward pass of the operation.
        """
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError

import math
from typing import Union

class gergen:

    __veri = None #A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None #Dimensions of the derivative (Shape)
    def str_helper(self,veri):
      if type(veri[0])==list:
        s="["
        for i in range(len(veri)-1):
          s+=self.str_helper(veri[i])
          s+="\n"
        s+=self.str_helper(veri[len(veri)-1])
        s+="]"
        return s
      else:
        return f"{veri}"

    def boy_helper(self,veri):
      if type(veri[0])==list:
        s=[len(veri)]
        s+=self.boy_helper(veri[0])
        return s
      else:
        return [len(veri)]

    def element_wise_helper(self,veri,other_veri,islem):
      s=[]
      if type(veri[0])==list:
        for i in range(len(veri)):
          s.append(self.element_wise_helper(veri[i],other_veri[i],islem))
        return s
      else:
        return [islem(a,b) for a,b in zip(veri,other_veri)]

    def carp(self,a,b):
      return a*b
    def bol(self,a,b):
      return a/b
    def cikar(self,a,b):
      return a-b
    def topla_be(self,a,b):
      return a+b
    def ussu(self,a,b):
      return a**b
    def dort_islem(self,veri,constant,islem):
      s=[]
      if type(veri[0])==list:
        for i in range(len(veri)):
          s.append(self.dort_islem(veri[i],constant,islem))
        return s
      else:
        return [islem(a,constant) for a in veri]


    def unilever(self,veri,islem):
      s=[]
      if type(veri[0])==list:
        for i in range(len(veri)):
          s.append(self.unilever(veri[i],islem))
        return s
      else:
        return [islem(a) for a in veri]

    def duz_helper(self,veri):
      if type(veri[0])==list:
        s=[]
        for i in range(len(veri)):
          s+=self.duz_helper(veri[i])
        return s
      else:
        return veri

    def index_helper(self,veri):
      if type(veri[0])==list:
        s=[]
        for i in range(len(veri)):
          s+=self.duz_helper(veri[i]).append(i)
        return s
      else:
        for i in range(len(veri)):
          s+=self.duz_helper(veri[i])
          return veri

    def help_trans(self,boy):
      k=1
      s=[]
      for i in range(len(boy)-1,-1,-1):
        k*=boy[i]
        s.append(k)
      s=s[:len(s)-1]
      if s==[]: s=[len(self.__veri)]
      return s

    def trans(self,help,duzlist):
      if len(help)>1:
        s=self.trans(help[1:],duzlist)
        return self.trans([help[0]],s)
      else:
        if(help[0]==1): #test et
           return duzlist
        new_list = [duzlist[i::help[0]] for i in range(help[0])]
        return new_list

    #def boyutlandir_h(boyut,elements):
    #  if len(boyut)>1:
    #    t=self.boyutlandir(boyut[1:],elements)
    #    return t.boyutlandir([boyut[0]],t)
    #  else:
    #    t = [elements[i:i+boyut[0]] for i in range(0,len(elements),boyut[0])]
    #    return t



    def __init__(self, veri=None):
      self.__veri=veri
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        self.__boyut=()
        self.D=self.__veri
      elif veri!=None:
        self.__boyut=tuple(self.boy_helper(veri))
        self.D=self.trans(self.help_trans(self.__boyut),self.duz_helper(self.__veri))

    def __getitem__(self, index):
      return gergen(self.__veri[index])

    def dimx(self):
      return 'x'.join(map(str, self.__boyut))

    def __str__(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return f"0 boyutlu gergen:\n {self.__veri}"
      a=f"{self.dimx()} boyutlu gergen:\n"
      return a+self.str_helper(self.__veri)

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        if type(other)==gergen:
          if isinstance(self.__veri,int) or isinstance(self.__veri,float):
            return other*self.__veri
          if self.__boyut!=other.boyut():
            raise ValueError('other boyut does not satisfies with self boyut')
          return gergen(self.element_wise_helper(self.__veri,other.__veri,self.carp))
        if type(other)==float or type(other)==int:
          return gergen(self.dort_islem(self.__veri,other,self.carp))
        raise TypeError('other mistype')


    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        if type(other)==gergen:
          if self.__boyut!=other.boyut:
            raise ValueError('other boyut does not satisfies with self boyut')
          return gergen(self.element_wise_helper(self.__veri,other.__veri,self.bol))
        if type(other)==float or type(other)==int:
          return gergen(self.dort_islem(self.__veri,other,self.bol))
        raise TypeError('other mistype')


    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        if type(other)==gergen:
          if isinstance(self.__veri,int) or isinstance(self.__veri,float):
            return other+self.__veri
          if self.__boyut!=other.boyut():
            raise ValueError('other boyut does not satisfies with self boyut')
          return gergen(self.element_wise_helper(self.__veri,other.__veri,self.topla_be))
        if type(other)==float or type(other)==int:
          return gergen(self.dort_islem(self.__veri,other,self.topla_be))
        raise TypeError('other mistype')

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        if type(other)==gergen:
          if self.__boyut!=other.boyut:
            raise ValueError('other boyut does not satisfies with self boyut')
          return gergen(self.element_wise_helper(self.__veri,other.__veri,self.cikar))
        if type(other)==float or type(other)==int:
          return gergen(self.dort_islem(self.__veri,other,self.cikar))
        raise TypeError('other mistype')

    def uzunluk(self):
      k=1
      for i in self.__boyut:
        k*=i
      return k

    def boyut(self):
      return self.__boyut

    def devrik(self):
      return gergen(self.D)

    def sin(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
          return gergen(math.sin(self.__veri))
      return gergen(self.unilever(self.__veri,math.sin))

    def cos(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
          return gergen(math.cos(self.__veri))
      return gergen(self.unilever(self.__veri,math.cos))

    def tan(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
          return gergen(math.tan(self.__veri))
      return gergen(self.unilever(self.__veri,math.tan))

    def us(self, n: int):
      if n<0:
        raise ValueError('n must be an integer')
      elif isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return gergen(self.__veri**n)
      else:
        return gergen(self.dort_islem(self.__veri,n,self.ussu))


    def log(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return gergen(math.log10(self.__veri))
      return gergen(self.unilever(self.__veri,math.log10))

    def ln(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return gergen(math.log(self.__veri))
      return gergen(self.unilever(self.__veri,math.log))

    def L1(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return math.abs(self.__veri)
      a=self.duz_helper(self.__veri)
      return math.sum(list(map(math.abs, a)))
    def L2(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return math.abs(self.__veri)
      a=self.duz_helper(self.__veri)
      return math.sqrt(math.sum([math.pow(i, 2) for i in a]))

    def Lp(self, p):
      if p<0:
        raise ValueError("misvalue. it must be non negative")
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return math.abs(self.__veri)
      a=self.duz_helper(self.__veri)
      return math.pow(math.sum([math.abs(math.pow(i, p)) for i in a]),1/p)

    def listeye(self):
      return self.__veri

    def duzlestir(self):
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return gergen([self._veri])
      return gergen(self.duz_helper(self.__veri))

    def boyutlandir(self, yeni_boyut):
      if type(yeni_boyut)!= tuple:
        raise TypeError(" mistype. it must be tuple")
      k=1
      for i in yeni_boyut:
        k*=i
      if(k!=self.uzunluk()):
        raise ValueError("misdimension")
      liste=list(yeni_boyut)
      a=self.help_trans(liste)
      if isinstance(self.__veri,int) or isinstance(self.__veri,float):
        return gergen(self.trans(a,[self.__veri]))
      return gergen(self.trans(a,self.duz_helper(self.__veri))).devrik()



    def ic_carpim(self, other):
      if type(other)!=type(self):
        raise TypeError("mistype")
      if len(self.boyut())!=1 or len(other.boyut())!=1:
        if len(self.boyut())>2 or len(other.boyut())>2 or len(self.boyut())==0 or len(other.boyut())==0:
          raise ValueError("misdimension")
        dim=(self.__boyut[0],other.boyut()[1])
        matlist=boyut_acan_dogal(dim,0,0)
        for i in range(self.__boyut[0]):
          for j in range(self.__boyut[1]):
            for k in range(other.boyut()[1]):
              matlist[i][j]+=self.__veri[i][k]*other.listeye()[k][j]
        return gergen(matlist)
      else:
        duzlist1=self.duz_helper(self.__veri)
        duzlist2=other.duz_helper(other.getveri())
        return math.sum([ a*b for a,b in zip(duzlist1, duzlist2) ])
    def dis_carpim(self, other):
      if type(other!=gergen):
        raise TypeError('other is not type of gergen')
      if len(self.__boyut)!=1 or len(other.boyut())!=1:
        raise ValueError("mis dimension")
        matlist=boyut_acan_dogal(dim,0,0)
        for i in range (self.__boyut[0]):
          for j in range(other.boyut()[0]):
            matlist[i][j]+=self.__veri[i][0]*other.isteye[j][0]
        return gergen(matlist)

    def help_topla(self,eksen,kendinden_sonra,duzlist):
      step=kendinden_sonra[eksen+1]
      start_incr=kendinden_sonra[eksen]
      #print("step",step,"start_inc",start_incr,"duzlist",duzlist)
      l=[]
      t=0
      for i in range(0,len(duzlist),start_incr):#start noktasını kaydırıyor
        for k in range(kendinden_sonra[eksen+1]):
          for j in range(0,start_incr,step):#
            #print("ikj",(i,k,j),i+j+k)
            t+=duzlist[i+j+k]
          #print("t=",t)
          l.append(t)
          t=0
      return gergen(l)



    def topla(self, eksen=None):
      if eksen==None:
        a=self.duz_helper(self.__veri)
        return gergen(math.sum(a))
      kendinden_sonra=self.help_trans(self.__boyut)
      kendinden_sonra.append(kendinden_sonra[-1]*self.__boyut[0])
      kendinden_sonra.reverse()
      kendinden_sonra.append(1)
      #print("kendinden sonra ",kendinden_sonra)
      duzlist=self.duz_helper(self.__veri)
      g=self.help_topla(eksen,kendinden_sonra,duzlist)
      #print("g burada",g)
      boyut_list=list(self.__boyut)
      del boyut_list[eksen]
      boyut_list=tuple(boyut_list)
      g=g.boyutlandir(boyut_list)
      #print("boyut_list:",boyut_list)
      #print("topla",g)
      return g

    def ortalama(self, eksen=None):
      g=self.topla(eksen)
      bol=self.__boyut[eksen]
      g=g/bol
      return g

"""## 2 Compare with NumPy"""

import numpy as np              # NumPy, for working with arrays/tensors
import time                     # For measuring time

"""**Example 1:**
Using rastgele_gercek(), generate two gergen objects with shapes (64,64) and calculate the a.ic carpim(b). Then, calculate the same function with NumPy and report the time and difference.
"""

def example_1():
    #Example 1
    boyut = (64,64)
    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)

    start = time.time()
    #TODO
    m=g1.ic_carpim(g2)
    #Apply given equation
    end = time.time()

    a=np.array(g1.listeye())
    b=np.array(g2.listeye())
    start_np = time.time()
    n=np.dot(a,b)
    #Apply the same equation for NumPy equivalent
    end_np = time.time()

    #TODO:
    #Compare if the two results are the same
    print(np.allclose(m.listeye(),n))
    #Report the time difference
    print("Time taken for gergen:", end-start)
    print("Time taken for numpy:", end_np-start_np)

"""**Example 2:**
Using rastgele_gercek(), generate three gergen’s a, b and c with shapes (4,16,16,16). Calculate given equation:

> (a×b + a×c + b×c).ortalama()

Report the time and whether there exists any computational difference in result with their NumPy equivalent.
"""

def example_2():
    #Example 2
    #TODO:
    boyut = (4,16,16,16)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)
    c = rastgele_gercek(boyut)
    start = time.time()
    #TODO
    m=a*b+a*c+b*c
    #Apply given equation
    end = time.time()

    d=np.array(a.listeye())
    e=np.array(b.listeye())
    f=np.array(c.listeye())
    start_np = time.time()
    n=d*e+d*f+e*f
    #Apply the same equation for NumPy equivalent
    end_np = time.time()

    #TODO:
    #Compare if the two results are the same
    print(np.allclose(m.listeye(),n))
    #Report the time difference
    print("Time taken for gergen:", end-start)
    print("Time taken for numpy:", end_np-start_np)


"""**Example 3**: Using rastgele_gercek(), generate three gergen’s a and b with shapes (3,64,64). Calculate given equation:


> $\frac{\ln\left(\left(\sin(a) + \cos(b)\right)^2\right)}{8}$


Report the time and whether there exists any computational difference in result with their NumPy equivalent.

"""

def example_3():
    #Example 2
    #TODO:
    boyut = (4,64,64)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)
    start = time.time()
    #TODO
    m=((a.sin()+b.cos()).us(2)).ln()/8
    #Apply given equation
    end = time.time()

    d=np.array(a.listeye())
    e=np.array(b.listeye())
    start_np = time.time()
    n=np.log(np.square(np.sin(d) + np.cos(e))) / 8
    #Apply the same equation for NumPy equivalent
    end_np = time.time()

    #TODO:
    #Compare if the two results are the same
    print(np.allclose(m.listeye(),n))
    #Report the time difference
    print("Time taken for gergen:", end-start)
    print("Time taken for numpy:", end_np-start_np)
