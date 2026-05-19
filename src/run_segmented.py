import argparse 
from pathlib import Path 
from .config import load_config, Paths 
from .sim import run_time_loop 
import jax.numpy as jnp

def compress_pod(f: jnp.ndarray, rank: int) -> jnp.ndarray:
    """Apply SVD and reconstruct the distribution f with rank r"""
    #SVD calculation
    U, s, VT = jnp.linalg.svd(f, full_matrices=False)
    
    #Truncation
    U_r = U[:, :rank]
    s_r = s[:rank]
    VT_r = VT[:rank, :]
    
    #reconstruction
    f_comp = (U_r * s_r) @ VT_r
    
    return f_comp   

def main():
    parser = argparse.ArgumentParser(description="Run PlasmaX simulation in segments.")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    parser.add_argument("--tend", type=float, default=100.0, help="Total simulation time")
    parser.add_argument("--rank", type=int, default=32, help="Rank for POD compression")
    args = parser.parse_args()
    
    #base configuration
    cfg = load_config(args.params)  
    
    dt_seg = cfg.io.dt_save if cfg.io.dt_save is not None else 5.0
    
    original_case = cfg.inicond.case 
    
    #creating specific folders to be able to compare the results
    if cfg.io.compression_method == "POD":
        segmented_case = f"{original_case}_segmented_POD_r{args.rank}"
    else: 
        segmented_case = f"{original_case}_segmented"
    
    cfg.inicond.case = segmented_case
    
    new_save_dir = cfg.paths.save_dir.parent / segmented_case
    cfg.paths = Paths.from_case(segmented_case, new_save_dir)
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    
    #clean up the old csv file if we restart the experiment from scratch
    diag_file = cfg.paths.data_dir / "diagnostics.csv" 
    if diag_file.exists():
        diag_file.unlink() 
        
    current_time = 0.0
    cfg.io.restart.enabled = False 
    cfg.io.restart.file = None 
    
    while current_time < args.tend - 1e-9:
        next_time = min(current_time + dt_seg, args.tend)
        
        print("\n" + "="*60)
        print(f"-> Running segment: t= {current_time:.2f} to t={next_time:.2f}")
        print("="*60 + "\n")
        
        #update of end time for this segment
        cfg.time.tend = next_time
        
        run_time_loop(cfg)
        
        current_time = next_time
        
        #interception and compression
        file_path = cfg.paths.data_dir / "f_final.npz"
        
        if cfg.io.compression_method == "POD":
            print(f"\n[POD] Applying compression (rank={args.rank}) to saved state...")
            data = jnp.load(file_path)
            f_full = data['f']
            t_saved = data['t']
            it_saved = data['it']
            
            f_comp = compress_pod(f_full, args.rank)
            
            #overwrite the saved state with the compressed version
            jnp.savez(file_path, f=f_comp, t=t_saved, it=it_saved)
            print(f"[POD] Done\n")
        
        cfg.io.restart.enabled = True
        cfg.io.restart.file = str(file_path)
        
        
        
    print(f"\n Segmented simulation finished successfully. Results saved in '{new_save_dir}' ")
    
if __name__ == "__main__":
    main()