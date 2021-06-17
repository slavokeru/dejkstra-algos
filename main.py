import heapq as hq
import math
import time
import random
from collections import deque
from typing import Dict
from timeit import Timer
########################################################################################################################
# 1: fibonachi heap


def dijkstra_fibonachi(G, s):
    n = len(G)
    visited = [False]*n
    ws = [math.inf]*n
    path = [None]*n
    queue = []
    ws[s] = 0
    hq.heappush(queue, (0, s))
    while len(queue) > 0:
        g, u = hq.heappop(queue)
        visited[u] = True
        for v, w in G[u]:
            if not visited[v]:
                f = g + w
                if f < ws[v]:
                    ws[v] = f
                    path[v] = u
                    hq.heappush(queue, (f, v))
    return path, ws
########################################################################################################################
# 2: simple, slow


def dijkstra_slow(N, S, matrix):
    valid = [True]*N
    w = [math.inf]*N
    w[S] = 0
    for i in range(N):
        min_w = math.inf
        ID_min_w = -1
        for j in range(N):
            if valid[j] and w[j] < min_w:
                min_w = w[j]
                ID_min_w = j
        for z in range(N):
            if w[ID_min_w] + matrix[ID_min_w][z] < w[z]:
                w[z] = w[ID_min_w] + matrix[ID_min_w][z]
        valid[ID_min_w] = False
    return w
########################################################################################################################
#3: A*


class Graph:
    def __init__(self, adjac_lis):
        self.adjac_lis = adjac_lis

    def get_neighbors(self, v):
        return self.adjac_lis[v]

    def heuristic(self, n):
        H = {str(i): 1 for i in range(length)}

        return H[n]

    def a_star_algorithm(self, start, stop):
        open_lst = set([start])
        closed_lst = set([])

        poo = {}
        poo[start] = 0

        par = {}
        par[start] = start

        while len(open_lst) > 0:
            n = None

            for v in open_lst:
                if n == None or poo[v] + self.heuristic(v) < poo[n] + self.heuristic(n):
                    n = v

            if n == None:
                return -1

            if n == stop:
                reconst_path = []

                while par[n] != n:
                    reconst_path.append(n)
                    n = par[n]

                reconst_path.append(start)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            for (m, w) in self.get_neighbors(n):
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = n
                    poo[m] = poo[n] + w

                else:
                    if poo[m] > poo[n] + w:
                        poo[m] = poo[n] + w
                        par[m] = n

                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)

            open_lst.remove(n)
            closed_lst.add(n)

        print('Path does not exist!')
        return None

########################################################################################################################
#main
length = 1000
""""""
print(length)
G3 = [[0 for j in range(length)]for i in range(length)]
for i in range(len(G3)):
    for j in range(len(G3)):
        if i < j:
            num = random.randint(0, 150)
            if num == 0:
                G3[i][j] = math.inf
            else:
                G3[i][j] = num
        elif i > j:
            G3[i][j] = G3[j][i]
        else:
            G3[i][j] = 0
"""
for i in range(len(G3)):
    print(i, ":", end=' ')
    for j in range(len(G3)):
        print((j, G3[i][j]), end=', ')
    print()
"""
t = Timer("""x.index(123)""", setup="""x = range(1000)""")
print(dijkstra_slow(length, 0, G3))
print(t.timeit())

G4 = [[]for i in range(length)]
for i in range(len(G3)):
    for j in range(len(G3)):
        if G3[i][j] == math.inf:
            pass
        else:
            G4[i].append((j, G3[i][j]))

t = Timer("""x.index(123)""", setup="""x = range(1000)""")
print(dijkstra_fibonachi(G4, 0)[1])
# print(time.time() - start_time)
print(t.timeit())
########################################################################################################################

adjac_lis: Dict = {}

for i in range(length):
    tmp_list = []
    for j in range(length):
        if i < j and G3[i][j] != 0 and G3[i][j] != math.inf:
            tmp_list.append((str(j), G3[i][j]))
    if i != length - 1:
        adjac_lis[str(i)] = tmp_list

# print(adjac_lis)
G5 = Graph(adjac_lis)

t = Timer("""x.index(123)""", setup="""x = range(1000)""")
list1 = G5.a_star_algorithm('0', str(length-1))
answer = 0
for i in range(len(list1) - 1):
    answer += G3[int(list1[i])][int(list1[i+1])]
print(list1, ", answer: ", answer)
print(t.timeit())
