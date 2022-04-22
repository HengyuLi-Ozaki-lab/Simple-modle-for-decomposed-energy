import os
import sys
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

symbol = []

with open(glob.glob(r"./*.dat")[0]) as f:

    lines = f.readlines()

    for i, line in enumerate(lines):
        if '<Atoms.SpeciesAndCoordinates' in line:
            begin = i
        elif 'Atoms.SpeciesAndCoordinates>' in line:
            over = i

    for line in lines[begin+1:over]:
        symbol.append(line.split()[1])

    for line in lines:
        if 'Atoms.Number' in line:
            atom_num = int(line.split()[1])
        if 'ML.Lammda_1' in line:
            lammda1 = line.split()[1]
        if 'ML.Lammda_2' in line:
            lammda2 = line.split()[1]
        if 'ML.Train_iter' in line:
            training_iter = int(line.split()[1])
        if 'ML.Correction_iter' in line:
            correct_iter = int(line.split()[1])
        if 'MD.maxIter' in line:
            max_iter = int(line.split()[1])
        if 'ML.Max_order' in line:
            max_order = int(line.split()[1])
        if 'ML.Min_order' in line:
            min_order = int(line.split()[1])

symbol_ana = defaultdict(list)

for i, element in enumerate(symbol):
    symbol_ana[element].append(i)

twobody_para = [[] for _ in range(atom_num)]
threebody_para = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.fitted_parameter")[0]) as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'atom' in line:
            atom = int(line.split()[1])-1
            if atom_num == 2:
                twobody_num = 1
            else:
                twobody_num = int((min_order-max_order-1+(max_order-min_order+1)**0.5*(max_order-min_order+1+120)**0.5)/(2*(max_order-min_order+1)))
            twobody_para[atom].append(list(map(float,lines[i+1].split()))[0:twobody_num*(max_order-min_order+1)])
            threebody_para[atom].append(list(map(float,lines[i+1].split()))[twobody_num*(max_order-min_order+1):])

distance = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.cal_dis")[0]) as f:
    lines = f.readlines()
    for i, line in enumerate(lines): 
        if 'MD' in line:
            for j, dis in enumerate(lines[i+1:i+1+atom_num]):
                distance[j].append(list(map(float,dis.split())))

twobody_energy = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.twobody_energy")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        for iter in line[7:].strip().replace('\n','').split(' ,'):
            if iter:
                twobody_energy[i].append(list(map(float,iter.split(' '))))

threebody_energy = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.threebody_energy")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        for iter in line[7:].strip().replace('\n','').split(' ,'):
            if iter:
                threebody_energy[i].append(list(map(float,iter.split(' '))))

ref_energy = []

with open(glob.glob(r"./*.ref_energy")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        ref_energy.append(list(map(float,line[7:].strip().replace('\n','').split(' '))))

fitted_energy = []

with open(glob.glob(r"./*.fitted_energy")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        fitted_energy.append(list(map(float,line[7:].strip().replace('\n','').split(' '))))

energy_error = []

with open(glob.glob(r"./*.energy_error")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        energy_error.append(list(map(float,line[7:].strip().replace('\n','').split(' '))))

total_force = []

with open(glob.glob(r"./*.total_force")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        total_force.append(list(map(float,line[7:].strip().replace('\n','').split(' '))))

numerical_force = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.numerical_force")[0]) as f:
    for line in f.readlines():
        if 'Atom' in line:
            atom = int(line.split()[1])-1
            numerical_force[atom].append(list(map(float,line.split()[2:])))

ref_force = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.ref_force")[0]) as f:
    for line in f.readlines():
        if 'Atom' in line:
            atom = int(line.split()[1])-1
            ref_force[atom].append(list(map(float,line.split()[2:])))

fitted_force = [[] for _ in range(atom_num)]

with open(glob.glob(r"./*.fitted_force")[0]) as f:
    for line in f.readlines():
        if 'Atom' in line:
            atom = int(line.split()[1])-1
            fitted_force[atom].append(list(map(float,line.split()[2:])))

force_error = []

with open(glob.glob(r"./*.force_error")[0]) as f:
    for i,line in enumerate(f.readlines()[1:]):
        force_error.append(list(map(float,line.strip().replace('\n','').split(' ')[2:])))

if not os.path.exists('./energy fig'):
    os.mkdir('./energy fig')
if not os.path.exists('./force fig'):
    os.mkdir('./force fig')
if not os.path.exists('./error fig'):
    os.mkdir('./error fig')

for atom, element in enumerate(symbol):
    fig, ax = plt.subplots(figsize=(16, 9))

    iter_ref = [_ for _ in range(len(twobody_energy[atom]))]
    l1 = ax.plot(iter_ref,[sum(i) for i in twobody_energy[atom]],label='twobody')
    l2 = ax.plot(iter_ref,[sum(i) for i in threebody_energy[atom]],label='threebody')
    l3 = ax.plot(iter_ref,np.array([sum(i) for i in twobody_energy[atom]])+np.array([sum(i) for i in threebody_energy[atom]]),label='total',marker="o")
    l4 = ax.plot(iter_ref,ref_energy[atom],label='reference',marker="x")
    ax.legend()
    plt.savefig('./energy fig/'+element+str(atom)+'_energy.jpg',dpi=300)

for atom, element in enumerate(symbol):
    for axis_num, axis in enumerate(['x','y','z']):
        fig, ax = plt.subplots(figsize=(16, 9))
        iter_ref = [_ for _ in range(len(twobody_energy[atom]))]
        ax.plot(iter_ref,np.array(ref_force[atom])[:,axis_num],label='ref',marker="o")
        ax.plot(iter_ref,np.array(numerical_force[atom])[:,axis_num],label='numerical',marker="x")
        ax.plot(iter_ref,np.array(fitted_force[atom])[:,axis_num],label='fitted',marker="*")
        ax.legend()
        plt.savefig('./force fig/'+element+str(atom)+'_force_'+axis+'.jpg',dpi=300)

ref = [_ for _ in range(len(energy_error[0]))]
ref_ana = [_ for _ in range(training_iter+correct_iter-2,len(energy_error[0]),correct_iter)]
ref_ana_correct = [_ for _ in range(training_iter-1,len(energy_error[0]),correct_iter)]

for element in symbol_ana.keys():
    fig, ax = plt.subplots(figsize=(16, 9))
    error_max = []
    error_min = []
    for i in symbol_ana[element]:
        loss_ana = []
        loss_ana_correct = []
        l1 = ax.scatter(ref,energy_error[i],s=3,label=symbol[i]+str(i+1),c='black')
        error_max.append(max(energy_error[i]))
        error_min.append(min(energy_error[i]))

        for j in range(training_iter+correct_iter-2,len(energy_error[0]),correct_iter):
            loss_ana.append(energy_error[i][j])

        for k in range(training_iter-1,len(energy_error[0]),correct_iter):
            loss_ana_correct.append(energy_error[i][k])

        l2 = ax.plot(ref_ana,loss_ana,label='Max '+symbol[i]+str(i+1),linewidth=2,c='r')
        l3 = ax.plot(ref_ana_correct,loss_ana_correct,label='Corrected '+symbol[i]+str(i+1),linewidth=2.5,c='b')

    ax.plot([training_iter-1,training_iter-1],[max(error_max),min(error_min)],c='black',linewidth=1,linestyle='--')

    ax.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Max order = '+str(max_order)+'\n'+'Lambda1 = '+lammda1+' Lambda2 = '+lammda2)
    plt.savefig('./error fig/'+element+'_energy'+'.jpg',dpi=300)

for element in symbol_ana.keys():
    fig, ax = plt.subplots(figsize=(16, 9))
    error_max = []
    error_min = []
    for i in symbol_ana[element]:
        loss_ana = []
        loss_ana_correct = []
        l1 = ax.scatter(ref,force_error[i],s=3,label=symbol[i]+str(i+1),c='black')
        error_max.append(max(force_error[i]))
        error_min.append(min(force_error[i]))

        for j in range(training_iter+correct_iter-2,len(energy_error[0]),correct_iter):
            loss_ana.append(energy_error[i][j])

        for k in range(training_iter-1,len(energy_error[0]),correct_iter):
            loss_ana_correct.append(energy_error[i][k])

        l2 = ax.plot(ref_ana,loss_ana,label='Max '+symbol[i]+str(i+1),linewidth=2,c='r')
        l3 = ax.plot(ref_ana_correct,loss_ana_correct,label='Corrected '+symbol[i]+str(i+1),linewidth=2.5,c='b')

    ax.plot([training_iter-1,training_iter-1],[max(error_max),min(error_min)],c='black',linewidth=1,linestyle='--')

    ax.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Max order = '+str(max_order)+'\n'+'Lambda1 = '+lammda1+' Lambda2 = '+lammda2)
    plt.savefig('./error fig/'+element+'_force'+'.jpg',dpi=300)

print("Successfully plot all figures")