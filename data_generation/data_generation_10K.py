import openmc
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import os
import warnings
import time
import shutil
from datetime import datetime, timedelta
import tempfile

# Path Configuration (Check your path)
os.environ.setdefault(
    'OPENMC_CROSS_SECTIONS',
    os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'),
)

warnings.filterwarnings("ignore")

# --- [Configuration Area] ---
OUTPUT_FILENAME = 'reactor_10k_final.csv'
CHECKPOINT_PREFIX = 'checkpoint_10k_'
NUM_PATTERNS = 10000
BATCH_SIZE = 1000  # [Modified] Save every 1,000 patterns (Approx. 7 hours)
NUM_PROCESSES = 6  # i5-12400 (6 Physical Cores)

# --- [Physical Constants] ---
pitch = 1.26
fuel_r = 0.4096
clad_id = 0.835
clad_od = 0.950
assembly_size = 17 * pitch


# --- [OpenMC Modeling Functions] ---
def create_materials():
    fuel = openmc.Material(name='UO2 3.1wt%')
    fuel.add_nuclide('U235', 0.031)
    fuel.add_nuclide('U238', 0.969)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.29769)

    gd = openmc.Material(name='Gd Poison 8wt%')
    gd.add_nuclide('U235', 0.031 * 0.92)
    gd.add_nuclide('U238', 0.969 * 0.92)
    gd.add_nuclide('O16', 2.0 * 0.92 + 3.0 * 0.08)
    gd.add_nuclide('Gd155', 0.08 * 0.148)
    gd.add_nuclide('Gd156', 0.08 * 0.199)
    gd.add_nuclide('Gd157', 0.08 * 0.156)
    gd.add_nuclide('Gd158', 0.08 * 0.249)
    gd.add_nuclide('Gd160', 0.08 * 0.218)
    gd.set_density('g/cm3', 10.5)

    # Effectively Water (Guide Tube fill)
    control = openmc.Material(name='B4C Control')
    control.add_nuclide('B10', 0.2 * 0.199)
    control.add_nuclide('B11', 0.2 * 0.801)
    control.add_nuclide('C12', 0.8)
    control.set_density('g/cm3', 2.52)

    zirc = openmc.Material(name='Zircaloy-4')
    zirc.add_element('Sn', 0.014)
    zirc.add_element('Fe', 0.0121)
    zirc.add_element('Cr', 0.0107)
    zirc.add_element('Ni', 0.0050)
    zirc.add_element('Zr', 0.9582)
    zirc.set_density('g/cm3', 6.55)

    water = openmc.Material(name='PWR Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.set_density('g/cm3', 0.701)

    materials = openmc.Materials([fuel, gd, control, zirc, water])
    materials.export_to_xml()
    return fuel, gd, control, zirc, water


def create_pin_universe(rod_type, fuel, gd, control, zirc, water):
    fuel_cyl = openmc.ZCylinder(r=fuel_r)
    clad_inner = openmc.ZCylinder(r=clad_id / 2)
    clad_outer = openmc.ZCylinder(r=clad_od / 2)
    box = openmc.model.RectangularPrism(width=pitch, height=pitch, boundary_type='reflective')

    # Logic: If rod_type is 2 (Guide Tube), fill with Water (ARO condition)
    fill = {0: fuel, 1: gd, 2: water}[rod_type]

    cells = [openmc.Cell(fill=fill, region=-fuel_cyl),
             openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
             openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer),
             openmc.Cell(fill=water, region=-box & +clad_outer)]
    return openmc.Universe(cells=cells)


def create_assembly(grid, fuel, gd, control, zirc, water):
    lattice = openmc.RectLattice()
    lattice.pitch = (pitch, pitch)
    lattice.lower_left = (-assembly_size / 2, -assembly_size / 2)
    lattice.dimension = (17, 17)
    lattice.universes = np.array(
        [[create_pin_universe(grid[i, j], fuel, gd, control, zirc, water) for j in range(17)] for i in range(17)])
    outer_box = openmc.model.RectangularPrism(assembly_size, assembly_size, boundary_type='reflective')
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])


def simulate_pattern(pattern_id):
    original_dir = os.getcwd()
    work_dir = tempfile.mkdtemp(prefix=f'run_{pattern_id}_', dir='/dev/shm')

    try:
        os.chdir(work_dir)

        # 1. Create Model
        fuel, gd, control, zirc, water = create_materials()
        np.random.seed(pattern_id)

        # Standard 17x17 Guide Tube Coordinates (0-indexed)
        gt_coords = [
            (2, 5), (2, 8), (2, 11),
            (3, 3), (3, 13),
            (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
            (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
            (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
            (13, 3), (13, 13),
            (14, 5), (14, 8), (14, 11)
        ]

        grid = np.zeros((17, 17), dtype=int)

        # Set Fixed Guide Tubes (Type 2)
        gt_indices = []
        for r, c in gt_coords:
            grid[r, c] = 2
            gt_indices.append(r * 17 + c)

        # Randomly place 16 Gd rods
        all_pos = list(range(289))
        fuel_pos = [i for i in all_pos if i not in gt_indices]

        gd_pos_selected = np.random.choice(fuel_pos, 16, replace=False)

        for idx in gd_pos_selected:
            r, c = divmod(idx, 17)
            grid[r, c] = 1

        grid_str = ''.join('f' if x == 0 else 'g' if x == 1 else 'c' for x in grid.flatten())

        geom = create_assembly(grid, fuel, gd, control, zirc, water)
        geom.export_to_xml()

        # 2. Settings (High-Fidelity)
        settings = openmc.Settings()
        settings.batches = 30  # High-Fi
        settings.inactive = 10  # High-Fi
        settings.particles = 20000  # High-Fi (20k particles)

        settings.seed = pattern_id + 1

        settings.source = openmc.IndependentSource(space=openmc.stats.Box([-assembly_size / 2, -assembly_size / 2, 0],
                                                                          [assembly_size / 2, assembly_size / 2, 1]))
        settings.export_to_xml()

        # 3. Tally
        tallies = openmc.Tallies()
        mesh = openmc.RegularMesh()
        mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-assembly_size / 2, -assembly_size / 2, 0)
        mesh.upper_right = (assembly_size / 2, assembly_size / 2, 1)
        power_tally = openmc.Tally(name='power')
        power_tally.filters = [openmc.MeshFilter(mesh)]
        power_tally.scores = ['fission']
        tallies.append(power_tally)
        tallies.export_to_xml()

        # 4. Run
        openmc.run(output=False)

        # 5. Parse Results
        with openmc.StatePoint('statepoint.30.h5') as sp:
            k_eff = sp.keff.nominal_value
            k_std = sp.keff.std_dev
            power = sp.get_tally(name='power').mean.ravel()

            fq = power.max() / power.mean() if power.mean() > 0 else 999.0
            power_2d = power.reshape(17, 17)
            channel_powers = power_2d.mean(axis=0)
            fdh = channel_powers.max() / channel_powers.mean() if channel_powers.mean() > 0 else 999.0

        return {'pattern_id': pattern_id, 'grid_str': grid_str, 'k_eff': round(k_eff, 5), 'k_std': round(k_std, 5),
                'fq': round(fq, 4), 'fdh': round(fdh, 4)}

    except Exception as e:
        print(f"❌ Error (ID: {pattern_id}): {e}")
        return None
    finally:
        os.chdir(original_dir)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)


def get_last_checkpoint_index():
    if not os.path.exists('.'): return 0
    checkpoints = [f for f in os.listdir('.') if f.startswith(CHECKPOINT_PREFIX) and f.endswith('.csv')]
    if not checkpoints: return 0
    numbers = []
    for f in checkpoints:
        try:
            num = int(f.replace(CHECKPOINT_PREFIX, '').replace('.csv', ''))
            numbers.append(num)
        except:
            pass
    return max(numbers) if numbers else 0


if __name__ == '__main__':
    print("🧹 Cleaning RAM disk...")
    try:
        for item in os.listdir('/dev/shm'):
            if item.startswith('run_'):
                shutil.rmtree(os.path.join('/dev/shm', item), ignore_errors=True)
    except:
        pass
    print("✅ Ready")

    restart_from = get_last_checkpoint_index()
    start_time = time.time()

    print(f"\n🚀 [High-Fi Mode] Starting 10K Data Generation")
    print(f"⚙️  Settings: batches=30, inactive=10, particles=20000 (Active: 400,000)")
    print(f"🎯 Target: {NUM_PATTERNS:,} (Resume from: {restart_from:,})")
    print("=" * 60)

    for batch_start in range(restart_from, NUM_PATTERNS, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, NUM_PATTERNS)

        batch_start_time = time.time()
        print(f"\n📦 Processing Batch: {batch_start} - {batch_end} (Size: {batch_end - batch_start})")

        with Pool(processes=NUM_PROCESSES) as pool:
            batch_results = list(
                tqdm(pool.imap(simulate_pattern, range(batch_start, batch_end)), total=batch_end - batch_start))

        # Save Checkpoint
        valid_results = [r for r in batch_results if r is not None]
        if valid_results:
            df = pd.DataFrame(valid_results)
            checkpoint_filename = f'{CHECKPOINT_PREFIX}{batch_end}.csv'
            df.to_csv(checkpoint_filename, index=False)
            print(f"💾 Checkpoint saved: {checkpoint_filename}")

            del df
            del valid_results
            del batch_results

        # Speed & ETA Calculation
        total_elapsed = time.time() - start_time
        batch_elapsed = time.time() - batch_start_time
        processed = batch_end - restart_from

        if processed > 0:
            avg_speed = total_elapsed / processed
            rate = 1 / avg_speed
            remaining_items = NUM_PATTERNS - batch_end
            remaining_time = remaining_items * avg_speed

            print(f"⚡ Batch Time: {str(timedelta(seconds=int(batch_elapsed)))}")
            print(f"⚡ Avg Speed: {avg_speed:.3f} sec/item ({rate:.1f} items/sec)")
            print(f"⏳ ETA: {str(timedelta(seconds=int(remaining_time)))}")

    # Final Merge
    print("\n" + "=" * 60)
    print("💾 Merging all checkpoints...")
    all_files = sorted([f for f in os.listdir('.') if f.startswith(CHECKPOINT_PREFIX)],
                       key=lambda x: int(x.replace(CHECKPOINT_PREFIX, '').replace('.csv', '')))

    if all_files:
        with open(OUTPUT_FILENAME, 'w') as outfile:
            for i, fname in enumerate(tqdm(all_files)):
                with open(fname, 'r') as infile:
                    if i != 0: infile.readline()
                    outfile.write(infile.read())
        print(f"🎉 All done! Final dataset saved to: {OUTPUT_FILENAME}")
    else:
        print("⚠️ No checkpoint files found to merge.")