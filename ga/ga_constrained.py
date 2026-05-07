import os
import csv
import numpy as np
import time
import tempfile
import shutil
import openmc
import concurrent.futures
import random
import argparse
from tqdm import tqdm

# --- [CLI Arguments] ---
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
SEED = args.seed

# --- [Seed Setting] ---
random.seed(SEED)
np.random.seed(SEED)

# --- [Configuration] ---
OUTPUT_FILE = f"./ga_baseline_single_gd16_seed{SEED}_results.csv"
os.environ.setdefault(
    'OPENMC_CROSS_SECTIONS',
    os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'),
)

POPULATION_SIZE = 40
NUM_GENERATIONS = 50  # 40 * 50 = 2000 simulations
MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 3

TARGET_K_EFF = 1.05
TARGET_FQ = 1.00
TARGET_FDH = 1.00

# 🚨 Gd 개수 고정 제약
FIXED_GD_COUNT = 16

# --- [Physics Setup] ---
pitch, fuel_r, clad_id, clad_od = 1.26, 0.4096, 0.835, 0.950
assembly_size = 17 * pitch

GT_COORDS = [
    (2, 5), (2, 8), (2, 11), (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)
]
GT_INDICES = set([r * 17 + c for r, c in GT_COORDS])
VALID_INDICES = [i for i in range(289) if i not in GT_INDICES]


def fix_grid_gt(grid_list):
    for i in range(289):
        if i in GT_INDICES:
            grid_list[i] = 'c'
    return grid_list


# 🚨 Gd 개수를 정확히 FIXED_GD_COUNT로 보정하는 함수
def enforce_gd_count(grid_list, target_count=FIXED_GD_COUNT):
    """Gd 개수가 target_count와 다르면 강제 보정"""
    g_positions = [i for i in VALID_INDICES if grid_list[i] == 'g']
    f_positions = [i for i in VALID_INDICES if grid_list[i] == 'f']
    current_g = len(g_positions)

    if current_g > target_count:
        # Gd가 많으면 랜덤으로 f로 전환
        to_remove = random.sample(g_positions, current_g - target_count)
        for idx in to_remove:
            grid_list[idx] = 'f'
    elif current_g < target_count:
        # Gd가 적으면 랜덤으로 g로 전환
        to_add = random.sample(f_positions, min(target_count - current_g, len(f_positions)))
        for idx in to_add:
            grid_list[idx] = 'g'

    return grid_list


# --- [OpenMC Engine] ---
def create_materials():
    fuel = openmc.Material(name='UO2')
    fuel.add_nuclide('U235', 0.031); fuel.add_nuclide('U238', 0.969); fuel.add_nuclide('O16', 2.0); fuel.set_density('g/cm3', 10.29769)
    gd = openmc.Material(name='Gd')
    gd.add_nuclide('U235', 0.031*0.92); gd.add_nuclide('U238', 0.969*0.92); gd.add_nuclide('O16', 2.0*0.92+3.0*0.08)
    gd.add_nuclide('Gd155', 0.08*0.148); gd.add_nuclide('Gd156', 0.08*0.199); gd.add_nuclide('Gd157', 0.08*0.156)
    gd.add_nuclide('Gd158', 0.08*0.249); gd.add_nuclide('Gd160', 0.08*0.218); gd.set_density('g/cm3', 10.5)
    zirc = openmc.Material(name='Zirc')
    for el, frac in [('Sn', 0.014), ('Fe', 0.0121), ('Cr', 0.0107), ('Ni', 0.0050), ('Zr', 0.9582)]: zirc.add_element(el, frac)
    zirc.set_density('g/cm3', 6.55)
    water = openmc.Material(name='Water')
    water.add_nuclide('H1', 2.0); water.add_nuclide('O16', 1.0); water.add_s_alpha_beta('c_H_in_H2O'); water.set_density('g/cm3', 0.701)
    materials = openmc.Materials([fuel, gd, zirc, water])
    materials.export_to_xml()
    return fuel, gd, zirc, water

def create_assembly(grid, fuel, gd, zirc, water):
    fuel_cyl = openmc.ZCylinder(r=fuel_r); clad_inner = openmc.ZCylinder(r=clad_id/2); clad_outer = openmc.ZCylinder(r=clad_od/2)
    box = openmc.model.RectangularPrism(width=pitch, height=pitch, boundary_type='reflective')

    def get_univ(rod_type):
        fill = {0: fuel, 1: gd, 2: water}[rod_type]
        cells = [openmc.Cell(fill=fill, region=-fuel_cyl), openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
                 openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer), openmc.Cell(fill=water, region=-box & +clad_outer)]
        return openmc.Universe(cells=cells)

    lattice = openmc.RectLattice(); lattice.pitch = (pitch, pitch); lattice.lower_left = (-assembly_size/2, -assembly_size/2); lattice.dimension = (17, 17)
    lattice.universes = np.array([[get_univ(grid[i, j]) for j in range(17)] for i in range(17)])
    outer_box = openmc.model.RectangularPrism(assembly_size, assembly_size, boundary_type='reflective')
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])

def evaluate_grid(grid_str):
    g_count = grid_str.count('g')
    work_dir = tempfile.mkdtemp(prefix='ga_run_', dir=tempfile.gettempdir())
    original_dir = os.getcwd()

    try:
        os.chdir(work_dir)
        fuel, gd, zirc, water = create_materials()
        grid = np.zeros((17, 17), dtype=int)
        for i, char in enumerate(grid_str):
            if char == 'g': grid[divmod(i, 17)] = 1
            elif char == 'c': grid[divmod(i, 17)] = 2

        create_assembly(grid, fuel, gd, zirc, water).export_to_xml()
        settings = openmc.Settings(); settings.batches = 30; settings.inactive = 10; settings.particles = 20000; settings.output = {'summary': False}
        settings.source = openmc.IndependentSource(space=openmc.stats.Box([-assembly_size/2, -assembly_size/2, 0], [assembly_size/2, assembly_size/2, 1]))
        settings.export_to_xml()

        tallies = openmc.Tallies(); mesh = openmc.RegularMesh(); mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-assembly_size/2, -assembly_size/2, 0); mesh.upper_right = (assembly_size/2, assembly_size/2, 1)
        pt = openmc.Tally(name='power'); pt.filters = [openmc.MeshFilter(mesh)]; pt.scores = ['fission']; tallies.append(pt)
        tallies.export_to_xml()

        openmc.run(output=False)
        with openmc.StatePoint('statepoint.30.h5') as sp:
            k_eff = sp.keff.nominal_value
            power = sp.get_tally(name='power').mean.ravel()
            fq = power.max() / power.mean() if power.mean() > 0 else 99.9
            fdh = power.reshape(17, 17).mean(axis=0).max() / power.reshape(17, 17).mean(axis=0).mean() if power.reshape(17, 17).mean(axis=0).mean() > 0 else 99.9

        fitness = 0.6 * fq + 0.4 * fdh + 100.0 * abs(k_eff - TARGET_K_EFF)
        return (fitness, k_eff, fq, fdh, g_count, grid_str)
    except Exception:
        return (99.9, 0.0, 99.9, 99.9, g_count, grid_str)
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


# --- [Genetic Algorithm Core (Gd=16 Constrained)] ---
def create_random_individual():
    ind = ['f'] * 289
    # 🚨 정확히 16개의 Gd만 배치
    g_positions = random.sample(VALID_INDICES, FIXED_GD_COUNT)
    for pos in g_positions:
        ind[pos] = 'g'
    return "".join(fix_grid_gt(ind))

def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    selected_indices = random.sample(range(len(population)), k)
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]

def uniform_crossover(parent1, parent2):
    child1, child2 = list(parent1), list(parent2)
    for i in VALID_INDICES:
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    # 🚨 교차 후 Gd 개수 보정
    child1 = enforce_gd_count(fix_grid_gt(child1))
    child2 = enforce_gd_count(fix_grid_gt(child2))
    return "".join(child1), "".join(child2)

def mutate(individual):
    ind = list(individual)
    # 🚨 Gd 위치를 swap 방식으로 돌연변이 (개수 보존)
    g_positions = [i for i in VALID_INDICES if ind[i] == 'g']
    f_positions = [i for i in VALID_INDICES if ind[i] == 'f']

    for g_pos in g_positions:
        if random.random() < MUTATION_RATE and f_positions:
            # g 하나를 f로, f 하나를 g로 swap → 총 개수 유지
            swap_target = random.choice(f_positions)
            ind[g_pos] = 'f'
            ind[swap_target] = 'g'
            f_positions.remove(swap_target)
            f_positions.append(g_pos)

    return "".join(fix_grid_gt(ind))


# --- [Main Execution] ---
if __name__ == "__main__":
    print("🧹 Cleaning temp files...")
    [shutil.rmtree(os.path.join(tempfile.gettempdir(), d)) for d in os.listdir(tempfile.gettempdir()) if d.startswith('ga_run_')]

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation", "individual_idx", "simulation_count", "elapsed_time_sec",
            "grid", "fitness", "k_eff", "fq", "fdh", "g_count"
        ])

    print(f"\n🚀 Starting GA Optimization (Target k={TARGET_K_EFF}, Gd FIXED={FIXED_GD_COUNT})")
    print(f"⚙️  Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS} (Total Sims: 2000)")

    population = [create_random_individual() for _ in range(POPULATION_SIZE)]
    sim_count = 0
    start_time = time.time()
    best_ever_fitness = 99.9
    best_ever_grid = ""

    for gen in range(1, NUM_GENERATIONS + 1):
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(POPULATION_SIZE, os.cpu_count() or 8)) as executor:
            results = list(tqdm(executor.map(evaluate_grid, population), total=POPULATION_SIZE, desc=f"Gen {gen}/{NUM_GENERATIONS}", leave=False))

        sim_count += POPULATION_SIZE

        fitnesses = [r[0] for r in results]
        grids = [r[5] for r in results]

        best_idx = np.argmin(fitnesses)
        best_gen_fitness = fitnesses[best_idx]
        best_gen_k = results[best_idx][1]
        best_gen_gd = results[best_idx][4]

        if best_gen_fitness < best_ever_fitness:
            best_ever_fitness = best_gen_fitness
            best_ever_grid = grids[best_idx]

        elapsed = time.time() - start_time

        # Log ALL individuals to CSV
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, r in enumerate(results):
                writer.writerow([
                    gen, i, sim_count, round(elapsed, 2),
                    r[5], round(r[0], 4), round(r[1], 5), round(r[2], 4), round(r[3], 4), r[4]
                ])

        valid_fitnesses = [f for f in fitnesses if f < 99.0]
        mean_fitness = np.mean(valid_fitnesses) if valid_fitnesses else 99.9
        print(f"Gen {gen:2d} | Best Fit: {best_gen_fitness:.3f} (k={best_gen_k:.4f}, Gd={best_gen_gd}) | Mean Fit: {mean_fitness:.3f} | Total Sims: {sim_count}")

        if gen == NUM_GENERATIONS:
            break

        new_population = []
        elite_indices = np.argsort(fitnesses)[:2]
        new_population.extend([grids[i] for i in elite_indices])

        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(grids, fitnesses)
            p2 = tournament_selection(grids, fitnesses)
            c1, c2 = uniform_crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(c2))

        population = new_population

    print("\n" + "=" * 60)
    print(f"🏆 GA Optimization Complete (Gd={FIXED_GD_COUNT} Constrained)! Total Simulations: {sim_count}")
    print(f"⭐ Best Ever Fitness: {best_ever_fitness:.4f}")
    print("=" * 60)