import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize.minpack import leastsq
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe
import shutil
import os
import fnmatch

working_directory = '/Users/dimitrygrebenyuk/Yandex.Disk.localized/Working/PDF/Refinements/ThO2/fits/dataset_03/th14plus3drop2/cont'
dataFile = 'Th22_03_spot1_0001_0000_summed_bsub_tmean_gs2.grf'
basename = 'th14plus3drop2'


def PDF_fit(structure, data, qmin, qmax, qdamp, qbroad,
            xmin, xmax, scale, d, zoomscale, run_number):
    cdsePDF = PDFContribution("CdSe")
    cdsePDF.loadData(data)
    cdsePDF.setCalculationRange(xmin=xmin, xmax=xmax, dx=0.01)
    cdseStructure = loadStructure(f'{structure}.xyz')
    cdsePDF.addStructure("CdSe", cdseStructure, periodic=False)
    cdseFit = FitRecipe()
    cdseFit.addContribution(cdsePDF)
    cdsePDF.CdSe.setQmin(qmin)
    cdsePDF.CdSe.setQmax(qmax)
    cdsePDF.CdSe.qdamp.value = qdamp
    cdsePDF.CdSe.qbroad.value = qbroad

    cdseFit.addVar(cdsePDF.scale, scale, tag='scale', fixed=False)
    cdseFit.addVar(cdsePDF.CdSe.delta1, d, tag='d1', fixed=True)
    zoomscale = cdseFit.newVar('zoomscale', value=zoomscale, tag='zoomscale')
    lattice = cdsePDF.CdSe.phase.getLattice()
    cdseFit.constrain(lattice.a, zoomscale)
    cdseFit.constrain(lattice.b, zoomscale)
    cdseFit.constrain(lattice.c, zoomscale)
    ThUiso = cdseFit.newVar("Ho_Uiso", value=3.12000000e-02,
                            tag='adp', fixed=True)
    OUiso = cdseFit.newVar("O_Uiso", value=1.16000000e-01,
                           tag='adp', fixed=True)
    atoms = cdsePDF.CdSe.phase.getScatterers()
    for atom in atoms:
        if atom.element == 'Th':
            cdseFit.constrain(atom.Uiso, ThUiso)
        elif atom.element == 'O':
            cdseFit.constrain(atom.Uiso, OUiso)

    cdseFit.clearFitHooks()

    leastsq(cdseFit.residual, cdseFit.values)
    cdseStructure.write(f'{structure}.xyz', "xyz")
    profile = cdseFit.CdSe.profile
    profile.savetxt(f'{structure}.fit')


def calculate_rmsd(profile_fit):
    data = pd.read_csv(profile_fit, delim_whitespace=True, index_col=False)
    data.columns = ['a', 'b', 'c', 'd', 'e']
    df = pd.DataFrame(data)
    experimental = df.c.to_numpy()
    calculated = df.b.to_numpy()
    return np.sqrt(((experimental-calculated)**2).mean())


def monte_carlo_step(structure, lamda, thth_threshold):

    with open(f'{structure}.xyz', 'r') as f:
        lines = f.readlines()

    displacement = np.random.uniform(-1, 1, 3*(len(lines)-3))
    norm = np.linalg.norm(displacement)
    displacement = displacement / norm * lamda
    displacement = np.array_split(displacement, (len(lines)-3))

    new_lines = [lines[0], lines[1]]
    for i, line in enumerate(lines[2:]):
        fields = line.split()
        k = i % len(displacement)
        line = (
            f'{fields[0]} {str(float(fields[1]) + displacement[k][0])} {str(float(fields[2]) + displacement[k][1])} {str(float(fields[3]) + displacement[k][2])}'
            + "\n"
        )
        new_lines.append(line)

    penalty = 0
    for i in range(2, len(lines)):
        if "Th" in lines[i]:
            fields_i = lines[i].split()
            for j in range(i+1, len(lines)):
                if "Th" in lines[j]:
                    fields_j = lines[j].split()
                    distance = np.linalg.norm(np.array([float(fields_i[1]), float(fields_i[2]), float(fields_i[3])])
                                             - np.array([float(fields_j[1]), float(fields_j[2]), float(fields_j[3])]))
                    if distance < thth_threshold:
                        penalty += (thth_threshold - distance)**2

    penalty_tho = 0
    for i in range(2, len(lines)):
        if "O" in lines[i]:
            fields_i = lines[i].split()
            for j in range(i+1, len(lines)):
                if "Th" in lines[j]:
                    fields_j = lines[j].split()
                    distance = np.linalg.norm(np.array([float(fields_i[1]), float(fields_i[2]), float(fields_i[3])])
                                             - np.array([float(fields_j[1]), float(fields_j[2]), float(fields_j[3])]))
                    if distance < 2.2 or distance > 2.6 and distance < 4:
                        penalty_tho += (2.424 - distance)**4
                    else:
                        penalty_tho += 0
    with open(f'{structure}.xyz', 'w') as f:
        f.writelines(new_lines)
    return penalty + 0.01 * penalty_tho


def monte_carlo(steps, structure, data, qmin, qmax, qdamp, qbroad,
                xmin, xmax, scale, d, zoomscale, initial_l, cooling_rate,
                run_number, thth_threshold, penalty_weight):
    PDF_fit(structure, data, qmin, qmax, qdamp, qbroad, xmin, xmax, scale, d, zoomscale, run_number)
    best_rmsd = calculate_rmsd(f'{structure}.fit')
    shutil.copyfile(f'{structure}.xyz', f'{structure}_MC.xyz')
    print(f'Initial RMSD value for run {run_number} is {best_rmsd:.4f}')
    lamda = initial_l
    rmsd_values = []
    l_values = []
    fitness_values = []

    for i in range(steps):
        penalty = monte_carlo_step(structure, lamda, thth_threshold)
        PDF_fit(structure, data, qmin, qmax, qdamp, qbroad, xmin, xmax, scale, d, zoomscale, run_number)
        rmsd = calculate_rmsd(f'{structure}.fit')
        rmsd_values.append(rmsd)
        l_values.append(lamda)
        fitness = rmsd + penalty * penalty_weight
        fitness_values.append(fitness)
        if fitness < best_rmsd:
            best_rmsd = fitness
            shutil.copyfile(f'{structure}.xyz', f'{structure}_MC.xyz')
            lamda = initial_l
        else:
            shutil.copyfile(f'{structure}_MC.xyz', f'{structure}.xyz')
            if rmsd > best_rmsd:
                lamda *= cooling_rate
    
        print(f'Step {i+1} of run {run_number+1}; best RMSD is {best_rmsd:.4f}', end='\r')
    print(f'Best RMSD value for run {run_number} is {best_rmsd:.4f}')

    return rmsd_values, l_values, fitness_values


def plot_rmsd_l_values(all_rmsd_values, all_l_values):
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlabel("MC Run Number", fontsize=14)
    ax1.set_ylabel("RMSD", fontsize=14)
    ax1.set_title("RMSD vs MC Run Number", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.scatter(range(1, len(all_rmsd_values)+1), all_rmsd_values, color='red', s=20)
    ax1.set_xlim(1, len(all_rmsd_values))
    ax1.set_ylim(min(all_rmsd_values), max(all_rmsd_values))
    plt.show()


def random_displacement(structure, initial_displacement):
    with open(f'{structure}.xyz', 'r') as f:
        lines = f.readlines()

    displacement = np.random.uniform(-1, 1, 3*(len(lines)-3))
    norm = np.linalg.norm(displacement)
    displacement = displacement / norm * initial_displacement
    displacement = np.array_split(displacement, (len(lines)-3))

    new_lines = [lines[0], lines[1]]
    for i, line in enumerate(lines[2:]):
        fields = line.split()
        k = i % len(displacement)
        line = (
            f'{fields[0]} {str(float(fields[1]) + displacement[k][0])} {str(float(fields[2]) + displacement[k][1])} {str(float(fields[3]) + displacement[k][2])}'
            + "\n"
        )
        new_lines.append(line)

    with open(f'{structure}.xyz', 'w') as f:
        f.writelines(new_lines)
    return


def monte_carlo_multiple_starts(n_runs, steps, structure,
                                data, qmin, qmax, qdamp, qbroad, xmin, xmax,
                                scale, d, zoomscale, initial_l, cooling_rate,
                                thth_threshold, penalty_weight):
    all_rmsd_values = []
    all_l_values = []
    for run in range(n_runs):
        shutil.copyfile(f'{structure}.xyz', f'{structure}_start{run}.xyz')
        structure_start = f'{structure}_start{run}'
        random_displacement(structure_start, 0.1)
        rmsd_values, l_values, fitness_values = monte_carlo(steps, structure_start, data, qmin, qmax, qdamp, qbroad, xmin, xmax, scale, d, zoomscale, initial_l, cooling_rate, run, thth_threshold, penalty_weight)
        all_rmsd_values.extend(rmsd_values)
        all_l_values.extend(l_values)
        best_rmsd = min(rmsd_values)
        best_structure_index = rmsd_values.index(best_rmsd)
        shutil.copyfile(f'{structure}_start{run}_MC.xyz', f'{structure}_MC_run{run}_bestRMSD_{best_rmsd:.4f}.xyz')
    return all_rmsd_values, all_l_values, fitness_values


def clean_files(pattern):
    for file in os.listdir('.'):
        if fnmatch.fnmatch(file, pattern):
            file_path = os.path.join('.', file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


os.chdir(working_directory)
clean_files('*start*')
clean_files('*run*')
clean_files(f'{basename}.xyz')
clean_files(f'{basename}_MC.xyz')
clean_files(f'{basename}.fit')
shutil.copyfile(f'{basename}_init.xyz', f'{basename}.xyz')

all_rmsd_values, all_l_values, fitness_values = monte_carlo_multiple_starts(
        n_runs=35, steps=3500, structure=basename,
        data=dataFile, qmin=1, qmax=21, qdamp=0.02, qbroad=0.02,
        xmin=1.7, xmax=20,
        scale=6.43991989e-0, d=2.669, zoomscale=1.035,
        initial_l=0.2, cooling_rate=0.9,
        thth_threshold=3.5, penalty_weight=0.1)

plot_rmsd_l_values(all_rmsd_values=all_rmsd_values, all_l_values=all_l_values)
