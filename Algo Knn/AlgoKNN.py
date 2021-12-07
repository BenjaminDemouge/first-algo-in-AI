# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:48:18 2021

@author: bende
"""

import random as rd

def distance(p1,p2):#Foncion permettant de calculer la distance entre 2 points (distance eulerienne)
    return sum([(i-j)**2  for i,j in zip(p1,p2)])**0.5

def import_(name):#permet de lire correctement un fichier
  with open(name,'r') as file:
    X = list()
    Y = list()
    data = list()
    for line in file.readlines():
      sep = line.split(',')
      X.append(list(map(float,sep[:6])))
      Y.append(list(sep[-1][5]))
      data = [x + [y] for x,y in zip(X,Y)]
      data = sorted(data, key = lambda x: x[1])
    return X,Y,data

def import_2(name):#permet de lire correctement le fichier finaltest
  with open(name,'r') as file:
    X = list()
    for line in file.readlines():
      sep = line.split(',')
      X.append(list(map(float,sep[:6])))
    return X


def train_test(data,coef):#permet de separer un dataset en deux partie 
  X,Y,data = import_(data)
  train = list()
  train_size = int(len(data)*coef)
  test = data
  i = 0
  while len(train)< train_size:
    i = rd.randint(0,len(data)-1)
    x = test.pop(i)
    train.append(x)
    i+=1

  return train, test

def train_(train,test,k):#permet de calculer la justesse de cette methode pour un certain k donnée
  C = []
  p = 0
  for t in test:
    C.append(prediction(train,k,t))
  for i in range(len(C)):
    if [C[i]] == test[i][6]:
      p+=1
  pourc = round(100*p/len(C),2)
  return pourc,k
  
def findK(data,n):#permet de determiner le meilleur k pour un dataset et une selection train/test donnée
  train, test = train_test(data,0.5)
  a = list()
  for i in range(2,n+2):
    a.append(train_(train,test,i))
  a = sorted(a, key = lambda x: x[0])
  return a[-1][-1]
    
def prediction2(data,k,z):#permet de predire la classe d'un point en fonction d'un dataset
  if type(data) == str:
    X,Y,data = import_(data)
  else:
    X = list()
    Y = list()
    for x in data:
      X.append(x[:6])
      Y.append(x[-1])
  distances = [(distance(z,x),y) for x,y in zip(X,Y)]
  distances = sorted(distances, key = lambda x: x[0])
  solution = distances[:k]

  t = []
  for i in range(k):
    t.append(solution[i][1])
  a,b,c,d,e=0,0,0,0,0
  for i in range(k):
    if solution[i][1] == ['A']:
      a+=1;
    if solution[i][1] == ['B']:
      b+=1;
    if solution[i][1] == ['C']:
      c+=1;
    if solution[i][1] == ['D']:
      d+=1;
    if solution[i][1] == ['E']:
      e+=1;
  a0 = round(100*a/k,2)
  b0 = round(100*b/k,2)
  c0 = round(100*c/k,2)
  d0 = round(100*d/k,2)
  e0 = round(100*e/k,2)
  res = sorted([a0,b0,c0,d0,e0])
  T=''
  if res[-1] == a0:
    T = 'classA'
  if res[-1] == b0:
    T = 'classB'
  if res[-1] == c0:
    T = 'classC'
  if res[-1] == d0:
    T = 'classD'
  if res[-1] == e0:
    T = 'classE'

  #s = f'class A à {a0}%',f'class B à {b0}%',f'class C à {c0}%',f'class D à {d0}%',f'class E à {e0}%',f'most likely to be class {T}'
  return T
def prediction(data,k,z):#permet de predire la classe d'un point en fonction du dataset
  if type(data) == str:
    X,Y,data = import_(data)
  else:
    X = list()
    Y = list()
    for x in data:
      X.append(x[:6])
      Y.append(x[-1])
  distances = [(distance(z,x),y) for x,y in zip(X,Y)]
  distances = sorted(distances, key = lambda x: x[0])
  solution = distances[:k]

  t = []
  for i in range(k):
    t.append(solution[i][1])
  a,b,c,d,e=0,0,0,0,0
  for i in range(k):
    if solution[i][1] == ['A']:
      a+=1;
    if solution[i][1] == ['B']:
      b+=1;
    if solution[i][1] == ['C']:
      c+=1;
    if solution[i][1] == ['D']:
      d+=1;
    if solution[i][1] == ['E']:
      e+=1;
  a0 = round(100*a/k,2)
  b0 = round(100*b/k,2)
  c0 = round(100*c/k,2)
  d0 = round(100*d/k,2)
  e0 = round(100*e/k,2)
  res = sorted([a0,b0,c0,d0,e0])
  T=''
  if res[-1] == a0:
    T = 'A'
  if res[-1] == b0:
    T = 'B'
  if res[-1] == c0:
    T = 'C'
  if res[-1] == d0:
    T = 'D'
  if res[-1] == e0:
    T = 'E'

  #s = f'class A à {a0}%',f'class B à {b0}%',f'class C à {c0}%',f'class D à {d0}%',f'class E à {e0}%',f'most likely to be class {T}'
  return T

def test(data):#permet de tester notre methode sur le fichier pretest
  X,Y,data = import_(data)
  p = 0
  classes = []
  k = findK('data.csv',10)
  for x in X:
    classes.append(prediction('data.csv',k,x))
  for i in range(len(data)):
    if [classes[i]] == Y[i]:
      p +=1
  print('A:', classes.count('A'),'B:', classes.count('B'),'C:', classes.count('C'),'D:', classes.count('D'),'E:', classes.count('E'))
  return classes, p/len(data)

def test2(data):#permet de tester notre methode sur le fichier finaltest
  X = import_2(data)
  classes = []
  k = findK('data.csv',10)
  for x in X:
    classes.append(prediction2('data.csv',k,x))
  print('A:', classes.count('classA'),'B:', classes.count('classB'),'C:', classes.count('classC'),'D:', classes.count('classD'),'E:', classes.count('classE'))
  return classes

p = test2('finalTest.csv')
with open("DEMOUGE_Benjamin.txt", "w") as filout:
  for c in p:
    filout.write("{}\n".format(c))