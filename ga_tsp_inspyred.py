#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Candidate solutions for the TSP can be most easily be represented as permutations 
of the list of city numbers (enumerating the order in which cities should be visited). 
For instance, if there are 5 cities, then a candidate solution might be [4, 1, 0, 2, 3]. 
This is how the TSP benchmark represents solutions.

Author: rex_huang61
E-mail: 442193160@qq.com
Github: https://github.com/zhiruihuang/Artificial_Intelligence/
Date: 2017/12/04
"""
print(__doc__)

from random import Random
from time import time
from time import sleep
import math
import inspyred
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


def plot(points, result):
    N = len(points)
    x = np.array( [points[i][0] for i in range(N)] )
    y = np.array( [points[i][1] for i in range(N)] )
    colors = np.array([0.8 for _ in range(N)])
    area = np.array([150 for _ in range(N)]) # 0 to 15 point radii
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    rs = np.array(result)
    for i in range(len(rs)-1):
      plt.plot(np.hstack((x[rs[i]], x[rs[i+1]])), np.hstack((y[rs[i]], y[rs[i+1]])), color='green')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()

def tsp_observer(population, num_generations, num_evaluations, args):
    global points
    try:
        canvas = args['canvas']
    except KeyError:
        canvas = Canvas(Tk(), bg='white', height=400, width=400)
        args['canvas'] = canvas
        
    result = population[0].candidate
    old_lines = canvas.find_withtag('line')
    for l in old_lines:
        canvas.delete(l)
    vert_radius = 5
    for (x, y) in points:
        x = x-85
        y = y-15
        y = 35-y
        canvas.create_oval(x*10-vert_radius, y*10-vert_radius, x*10+vert_radius, y*10+vert_radius, fill='blue', tags='vert')
    rs = result
    for i in range(len(rs)-1):
        x1 = points[rs[i]][0]-85
        y1 = points[rs[i]][1]-15
        y1 = 35-y1
        x2 = points[rs[i+1]][0]-85
        y2 = points[rs[i+1]][1]-15
        y2 = 35-y2
        canvas.create_line(x1*10, y1*10, x2*10, y2*10, fill="red", tags="line")
    canvas.pack()
    canvas.update()
    print('{0} generations, Solution: : {1}'.format(num_generations, 1/population[0].fitness))
    sleep(0.05)


def main(prng=None, display=False):
    if prng is None:
        prng = Random() # 随机数发生器
        prng.seed(time()) # 时间种子

    global points
    points = [(116.46, 39.92), 
              (117.2,39.13), 
              (121.48, 31.22), 
              (106.54, 29.59), 
              (91.11, 29.97), 
              (87.68, 43.77), 
              (106.27, 38.47), 
              (111.65, 40.82), 
              (108.33, 22.84), 
              (126.63, 45.75), 
              (125.35, 43.88), 
              (123.38, 41.8), 
              (114.48, 38.03), 
              (112.53, 37.87), 
              (101.74, 36.56), 
              (117,36.65), 
              (113.6,34.76), 
              (118.78, 32.04), 
              (117.27, 31.86), 
              (120.19, 30.26), 
              (119.3, 26.08), 
              (115.89, 28.68), 
              (113, 28.21), 
              (114.31, 30.52), 
              (113.23, 23.16), 
              (121.5, 25.05), 
              (110.35, 20.02), 
              (103.73, 36.03), 
              (108.95, 34.27), 
              (104.06, 30.67), 
              (106.71, 26.57), 
              (102.73, 25.04), 
              (114.1, 22.2), 
              (113.33, 22.13)] # 城市坐标
    N = len(points)
    weights = [[0 for _ in range(N)] for _ in range(N)] # 城市距离
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            weights[i][j] = math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2) # 欧氏距离
    problem = inspyred.benchmarks.TSP(weights)
    ea = inspyred.ec.GA(prng)
    ea.selector = inspyred.ec.selectors.tournament_selection
    ea.variator = [inspyred.ec.variators.partially_matched_crossover, 
                   inspyred.ec.variators.inversion_mutation]
    ea.replacer = inspyred.ec.replacers.generational_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    window = Tk()
    window.title('TSP')
    can = Canvas(window, bg='white', height=350, width=450)
    can.pack()
    ea.observer = tsp_observer
    final_pop = ea.evolve(generator=problem.generator, # 发生器
                          evaluator=problem.evaluator, # 评估器
                          pop_size=100, # 个体数目
                          maximize=problem.maximize, # 是否求最值
                          bounder=problem.bounder,  # 是否有界
                          max_generations=100, # 遗传代数
                          tournament_size=34, # 样本个数
                          num_selected=100, # 每代选择个数
                          num_elites=1, # 精英个数
                          crossover_rate=1.0, # 交叉概率
                          mutation_rate=0.1, # 变异概率
                          canvas=can
                          )
    if display:
        best = max(ea.population)
        print('Best Solution: {0}: {1}'.format(str(best.candidate), 1/best.fitness))
        plot(points, best.candidate) # 散点图+线段
    return ea


if __name__ == '__main__':
    main(display=True)
