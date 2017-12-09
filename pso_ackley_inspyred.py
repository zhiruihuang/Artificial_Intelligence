#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
In this example, a PSO is used to evolve a solution to the Ackley benchmark.
In mathematical optimization, the Ackley function is a non-convex function 
used as a performance test problem for optimization algorithms. 
It was proposed by David Ackley in his 1987 PhD Disertation.
Its global optimum point is: f(0, 0)=0. 

Author: rex_huang61
E-mail: 442193160@qq.com
Github: https://github.com/zhiruihuang/Artificial_Intelligence/
Date: 2017/12/09
"""
print(__doc__)

from time import time
from time import sleep
from random import Random
import inspyred
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tkinter import *


def plot():
    fig = plt.figure()
    ax = Axes3D(fig)
    range = 5
    x1 = np.arange(-range, range, range/50)
    x2 = np.arange(-range, range, range/50)
    x1, x2 = np.meshgrid(x1, x2)
    y = -20 * np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2))) - np.exp(0.5*(np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2))) + np.exp(1) + 20
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title("Ackley Function")
    ax.set_xlim(-range, range)
    ax.set_ylim(-range, range)
    ax.set_xticks([-range, 0, range])
    ax.set_yticks([-range, 0, range])
    ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap='copper')
    # ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()


def pso_observer(population, num_generations, num_evaluations, args):
    try:
        canvas = args['canvas']
    except KeyError:
        canvas = Canvas(Tk(), bg='white', height=400, width=400)
        args['canvas'] = canvas

    old_verts = canvas.find_withtag('vert')
    for v in old_verts:
        canvas.delete(v)
    vert_radius = 3
    for c in population:
        (x, y) = c.candidate
        canvas.create_oval(x*1-vert_radius+150, y*1-vert_radius+150, x*1+vert_radius+150, y*1+vert_radius+150, fill='red', tags='vert')
    canvas.pack()
    canvas.update()
    print('{0} generations, Solution: : {1}'.format(num_generations, population[0].fitness))
    sleep(0.05)


def main(prng=None, display=False):
    if prng is None:
        prng = Random()
        prng.seed(time()) 
    
    problem = inspyred.benchmarks.Ackley(2)
    ea = inspyred.swarm.PSO(prng)
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    ea.topology = inspyred.swarm.topologies.ring_topology

    window = Tk()
    window.title('PSO')
    can = Canvas(window, bg='white', height=300, width=300)
    can.pack()
    ea.observer = pso_observer

    final_pop = ea.evolve(generator=problem.generator, # 发生器
                          evaluator=problem.evaluator, # 评估器
                          pop_size=100, # 个体数目
                          bounder=problem.bounder, # 是否求最值
                          maximize=problem.maximize, # 是否有界
                          max_evaluations=30000, # 最大评估数
                          neighborhood_size=5,
                          canvas=can
                          )

    if display:
        best = max(final_pop) 
        print('Best Solution: \n{0}'.format(str(best)))
        print('Global Optimum: \n{0}'.format(str(problem.global_optimum)))
    return ea


if __name__ == '__main__':
    plt.ion()
    plot()
    main(display=True)
    plt.ioff()
    plt.show()
