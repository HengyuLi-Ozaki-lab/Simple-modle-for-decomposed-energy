import matplotlib.pyplot as plt
import collections
import os

for file in os.listdir('.'):
    if os.path.splitext(file)[1] == '.error':
        error_file = file
    if os.path.splitext(file)[1] == '.dat':
        input_file = file

symbol = []
ML_info = []

with open(input_file) as f:

    lines = f.readlines()

    for i, line in enumerate(lines):
        if '<Atoms.SpeciesAndCoordinates' in line:
            begin = i
        elif 'Atoms.SpeciesAndCoordinates>' in line:
            over = i

    for line in lines[begin+1:over]:
        symbol.append(line.split()[1])

    for line in lines:

        if 'ML.Train_iter' in line:
            ML_info.append(int(line.split()[1]))
        if 'ML.Lammda_1' in line:
            ML_info.append(line.split()[1])
        if 'ML.Lammda_2' in line:
            ML_info.append(line.split()[1])
        if 'ML.Correction_iter' in line:
            ML_info.append(int(line.split()[1]))
        if 'ML.Max_order' in line:
            ML_info.append(line.split()[1])

symbol_ana = collections.defaultdict(list)

for i, element in enumerate(symbol):
    symbol_ana[element].append(i)

loss = []

with open(error_file,"r") as f:
    for line in f.readlines()[1:]:
        loss.append(list(map(float,line.strip().replace('\n','').split(' ')[2:])))

ref = [_ for _ in range(len(loss[0]))]
ref_ana = [_ for _ in range(ML_info[1]-2,len(loss[0]),ML_info[2])]

for element in symbol_ana.keys():
    fig, ax = plt.subplots(figsize=(16, 9))
    error_max = []
    error_min = []
    for i in symbol_ana[element]:
        loss_ana = []
        ax.scatter(ref,loss[i],s=2,label=symbol[i]+str(i+1))
        error_max.append(max(loss[i]))
        error_min.append(min(loss[i]))

        for j in range(ML_info[1]-2,len(loss[0]),ML_info[2]):
            loss_ana.append(loss[i][j])

        ax.plot(ref_ana,loss_ana,linewidth=2.5)

    ax.plot([ML_info[1]-1,ML_info[1]-1],[max(error_max),min(error_min)],c='black',linewidth=1,linestyle='--')

    ax.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Max order = '+ML_info[0]+'\n'+'Lammda1 = '+ML_info[3]+' Lammda1 = '+ML_info[4])
    plt.savefig('./'+element+'.jpg',dpi=300)