import os
import numpy as np
import tempfile
import shutil
import openmc
import time

# --- [Configuration] ---
os.environ.setdefault(
    'OPENMC_CROSS_SECTIONS',
    os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'),
)

TARGET_K_EFF = 1.05

# --- [Physics Setup] ---
pitch, fuel_r, clad_id, clad_od = 1.26, 0.4096, 0.835, 0.950
assembly_size = 17 * pitch
GT_COORDS = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13), (5, 2), (5, 5), (5, 8), (5, 11), (5, 14), (8, 2), (8, 5), (8, 8), (8, 11), (8, 14), (11, 2), (11, 5), (11, 8), (11, 11), (11, 14), (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)]
GT_INDICES = set([r * 17 + c for r, c in GT_COORDS])

# ---------------------------------------------------------
# 🏭 인간 전문가의 완벽한 대칭 패턴 (Industry Standard)
# ---------------------------------------------------------
def create_symmetric_grid(gd_coords):
    grid = ['f'] * 289
    for r, c in GT_COORDS:
        grid[r * 17 + c] = 'c'
    for r, c in gd_coords:
        grid[r * 17 + c] = 'g'
    return "".join(grid)

# 1. NuScale/PWR Standard 16-Gd Pattern (Octant Symmetry)
GD_16_COORDS = [
    (3, 6), (3, 10), (6, 3), (6, 13), (10, 3), (10, 13), (13, 6), (13, 10), # 십자가 바깥쪽
    (4, 4), (4, 12), (12, 4), (12, 12),                                     # 모서리 안쪽
    (7, 7), (7, 9), (9, 7), (9, 9)                                          # 중앙 코어 주변
]
nuscale_16_grid = create_symmetric_grid(GD_16_COORDS)

# 2. NuScale/PWR Standard 24-Gd Pattern (Octant Symmetry)
GD_24_COORDS = GD_16_COORDS + [
    (2, 2), (2, 14), (14, 2), (14, 14),                                     # 최외곽 모서리
    (6, 6), (6, 10), (10, 6), (10, 10)                                      # 중앙 십자가 추가
]
nuscale_24_grid = create_symmetric_grid(GD_24_COORDS)

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

def create_assembly(grid_str, fuel, gd, zirc, water):
    fuel_cyl = openmc.ZCylinder(r=fuel_r); clad_inner = openmc.ZCylinder(r=clad_id/2); clad_outer = openmc.ZCylinder(r=clad_od/2)
    box = openmc.model.RectangularPrism(width=pitch, height=pitch, boundary_type='reflective')
    
    def get_univ(char):
        fill = {'f': fuel, 'g': gd, 'c': water}[char]
        cells = [openmc.Cell(fill=fill, region=-fuel_cyl), openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
                 openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer), openmc.Cell(fill=water, region=-box & +clad_outer)]
        return openmc.Universe(cells=cells)
    
    lattice = openmc.RectLattice(); lattice.pitch = (pitch, pitch); lattice.lower_left = (-assembly_size/2, -assembly_size/2); lattice.dimension = (17, 17)
    lattice.universes = np.array([[get_univ(grid_str[i * 17 + j]) for j in range(17)] for i in range(17)])
    outer_box = openmc.model.RectangularPrism(assembly_size, assembly_size, boundary_type='reflective')
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])

def evaluate_baseline(name, grid_str):
    print(f"\n🚀 Evaluating {name} Baseline...")
    print(f"Grid Layout (Gd={grid_str.count('g')}):")
    for i in range(17):
        print(" ".join(grid_str[i*17:(i+1)*17]))
        
    work_dir = tempfile.mkdtemp(prefix='nuscale_run_', dir=tempfile.gettempdir())
    original_dir = os.getcwd()

    try:
        os.chdir(work_dir)
        fuel, gd, zirc, water = create_materials()
        create_assembly(grid_str, fuel, gd, zirc, water).export_to_xml()
        
        settings = openmc.Settings(); settings.batches = 30; settings.inactive = 10; settings.particles = 20000; settings.output = {'summary': False}
        settings.source = openmc.IndependentSource(space=openmc.stats.Box([-assembly_size/2, -assembly_size/2, 0], [assembly_size/2, assembly_size/2, 1]))
        settings.export_to_xml()
        
        tallies = openmc.Tallies(); mesh = openmc.RegularMesh(); mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-assembly_size/2, -assembly_size/2, 0); mesh.upper_right = (assembly_size/2, assembly_size/2, 1)
        pt = openmc.Tally(name='power'); pt.filters = [openmc.MeshFilter(mesh)]; pt.scores = ['fission']; tallies.append(pt)
        tallies.export_to_xml()

        start_time = time.time()
        openmc.run(output=False)
        
        with openmc.StatePoint('statepoint.30.h5') as sp:
            k_eff = sp.keff.nominal_value
            power = sp.get_tally(name='power').mean.ravel()
            fq = power.max() / power.mean()
            fdh = power.reshape(17, 17).mean(axis=0).max() / power.reshape(17, 17).mean(axis=0).mean()

        fitness = 0.6 * fq + 0.4 * fdh + 100.0 * abs(k_eff - TARGET_K_EFF)
        
        print("\n" + "="*50)
        print(f"🏆 {name} Results:")
        print(f"   k_eff   : {k_eff:.5f} (Target: {TARGET_K_EFF})")
        print(f"   F_q     : {fq:.4f}")
        print(f"   F_dH    : {fdh:.4f}")
        print(f"   Fitness : {fitness:.4f}")
        print("="*50)
        
        return {
            'name': name,
            'k_eff': k_eff,
            'fq': fq,
            'fdh': fdh,
            'gd_count': grid_str.count('g'),
            'fitness': fitness,
            'grid': grid_str
        }
        
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    import csv
    
    res_16 = evaluate_baseline("NuScale Standard 16-Gd", nuscale_16_grid)
    res_24 = evaluate_baseline("NuScale Standard 24-Gd", nuscale_24_grid)
    
    results = [res_16, res_24]
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nuscale_baseline_results.csv")
    
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\nResults successfully saved to: {save_path}")