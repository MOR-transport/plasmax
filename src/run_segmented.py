import argparse 
from pathlib import Path 
from .config import load_config, Paths 
from .sim import run_time_loop 

def main():
    parser = argparse.ArgumentParser(description="Run PlasmaX simulation in segments.")
    parser.add_argument("--params", type=str, required=True, help="Base yaml config file")
    parser.add_argument("--tend", type=float, default=100.0, help="Total simulation time")
    parser.add_argument("--dt-seg", type=float, default=5.0, help="Time duration of each segment")
    args = parser.parse_args()
    
    #base configuration
    cfg = load_config(args.params)  
    
    #creating a segmented case to separate the results
    original_case = cfg.inicond.case 
    segmented_case = f"{original_case}_segmented"
    cfg.inicond.case = segmented_case #the results will go in results/two_stream_segmented/
    
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
        next_time = min(current_time + args.dt_seg, args.tend)
        
        print("\n" + "="*60)
        print(f"-> Running segment: t= {current_time:.2f} to t={next_time:.2f}")
        print("="*60 + "\n")
        
        #update of end time for this segment
        cfg.time.tend = next_time
        
        run_time_loop(cfg)
        
        current_time = next_time
        cfg.io.restart.enabled = True 
        
        cfg.io.restart.file = str(cfg.paths.data_dir / "f_final.npz")
        
    print(f"\n Segmented simulation finished successfully. Results saved in '{new_save_dir}' ")
    
if __name__ == "__main__":
    main()