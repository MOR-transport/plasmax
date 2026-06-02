import argparse 
from pathlib import Path 
from .config import load_config, Paths 
from .sim import run_time_loop 
import jax.numpy as jnp

import pickle 
from .compression import compress_inr, compress_pod, AVAILABLE_INR_ARCHS
def main():
    parser = argparse.ArgumentParser(description="Run PlasmaX simulation in segments.")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    parser.add_argument("--tend", type=float, default=None, help="Total simulation time")
    parser.add_argument("--rank", type=int, default=32, help="Rank for POD compression")
    parser.add_argument("--compression", type=str, choices=["POD", "INR", "NONE"], default=None, help="Override yaml compression method")
    parser.add_argument("--arch", type=str, default="periodic_mlp_64", choices=AVAILABLE_INR_ARCHS, help="INR architecture (if compression method is INR)")
    args = parser.parse_args()
    
    #base configuration
    cfg = load_config(args.params)  
    final_time = args.tend if args.tend is not None else cfg.time.tend
    dt_seg = cfg.io.dt_save if cfg.io.dt_save is not None else 5.0
    
    original_case = cfg.inicond.case 
    #creating specific folders depending on the compression method and rank (for POD)
    compression = args.compression if args.compression is not None else cfg.io.compression_method
    cfg.inicond.case = f"{original_case}_segmented"
    case_root = Path("results") / f"case_{original_case}"
    
    if compression == "POD":
        folder_name = f"r{args.rank}"
        new_save_dir = case_root / "segmented" / "POD" / folder_name
    elif compression == "INR":
        folder_name = f"inr_{args.arch}"
        new_save_dir = case_root / "segmented" / "INR" / folder_name
    else: 
        folder_name = "NO_COMPRESSION"
        new_save_dir = case_root / "segmented" / folder_name
    
    #paths update in the configuration
    cfg.paths = Paths.from_case(cfg.inicond.case, new_save_dir)
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    
    #clean up the old csv file if we restart the experiment from scratch
    diag_file = cfg.paths.data_dir / "diagnostics.csv" 
    if diag_file.exists():
        diag_file.unlink() 
        
    current_time = 0.0
    cfg.io.restart.enabled = False 
    cfg.io.restart.file = None 
    current_nn_params = None
    
    while current_time < final_time - 1e-9:
        next_time = min(current_time + dt_seg, final_time)
        
        print("\n" + "="*60)
        print(f"-> Running segment: t= {current_time:.2f} to t={next_time:.2f}")
        print("="*60 + "\n")
        
        #update of end time for this segment
        cfg.time.tend = next_time
        run_time_loop(cfg)
        current_time = next_time
        
        #interception and compression
        file_path = cfg.paths.data_dir / "f_final.npz"
        data = jnp.load(file_path)
        f_full = data['f']
        t_saved = data['t']
        it_saved = data['it']
        
        plot_dir = cfg.paths.save_dir.parent / folder_name / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        if compression == "POD":
            print(f"\n[POD] Applying compression (rank={args.rank}) to saved state...")
            f_comp = compress_pod(f_full, args.rank, current_time, plot_dir, cfg.paths.data_dir)
            #overwrite the saved state with the compressed version
            jnp.savez(file_path, f=f_comp, t=t_saved, it=it_saved)
            print(f"[POD] Done\n")
            
        elif compression == "INR":
            print(f"\n[INR] Applying Neural Network compression...")
            f_comp, current_nn_params = compress_inr(
                f_full=f_full,
                grid_X=cfg.grid.X,
                grid_V=cfg.grid.V,
                lx=cfg.grid.lx,
                lv=cfg.grid.lv,
                current_time=current_time,
                data_dir=cfg.paths.data_dir,
                arch=args.arch,
                params_init=current_nn_params
            )
            
            jnp.savez(file_path, f=f_comp, t=t_saved, it=it_saved)
            weights_path = cfg.paths.data_dir / f"nn_weights_t{current_time:05.2f}.pkl"
            with open(weights_path, "wb") as f_weights:
                pickle.dump(current_nn_params, f_weights)
                
            print(f"[INR] Done (Weights saved to {weights_path.name})\n")
        
        cfg.io.restart.enabled = True
        cfg.io.restart.file = str(file_path)
        
    print(f"\n Segmented simulation finished successfully. Results saved in '{new_save_dir}' ")
    
if __name__ == "__main__":
    main()