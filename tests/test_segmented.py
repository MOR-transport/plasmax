import csv 
from pathlib import Path 
from src.config import load_config, Paths
from src.sim import run_time_loop 

def test_segmented_vs_continuous(tmp_path: Path):
    """Test that continuous simulation and segmented simulation produce exactly the same results (diagnostics.csv)."""
    tend = 1.0 
    dt_seg = 0.25 
    
    #continuous simulation
    cfg_cont = load_config("params/two_stream.yaml")
    cfg_cont.time.tend = tend 
    cfg_cont.time.plot_freq = 0
    
    cfg_cont.paths = Paths.from_case("continuous", tmp_path / "continuous")
    cfg_cont.paths.data_dir.mkdir(parents=True, exist_ok=True)
    
    run_time_loop(cfg_cont)
    
    #segmented simulation
    cfg_seg = load_config("params/two_stream.yaml")
    cfg_seg.time.plot_freq = 0
    cfg_seg.paths = Paths.from_case("segmented", tmp_path / "segmented")
    cfg_seg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    
    current_time = 0.0
    cfg_seg.io.restart.enabled = False 
    cfg_seg.io.restart.file = None 
    
    while current_time < tend - 1e-9:
        next_time = min(current_time + dt_seg, tend)
        cfg_seg.time.tend = next_time
        
        run_time_loop(cfg_seg)
        
        current_time = next_time
        cfg_seg.io.restart.enabled = True
        cfg_seg.io.restart.file = str(cfg_seg.paths.data_dir / "f_final.npz")
        
    #compare diagnostics.csv files
    diag_cont = cfg_cont.paths.data_dir / "diagnostics.csv"
    diag_seg = cfg_seg.paths.data_dir / "diagnostics.csv"

    assert diag_cont.exists(), "Le fichier CSV continu n'a pas été créé."
    assert diag_seg.exists(), "Le fichier CSV segmenté n'a pas été créé."
    
    with open(diag_cont, "r") as f_cont, open(diag_seg, "r") as f_seg:
        reader_cont = list(csv.reader(f_cont))
        reader_seg = list(csv.reader(f_seg))
        
        #check 1 : same number of lines 
        assert len(reader_cont) == len(reader_seg), "CSV files have not the same number of lines."
        
        #check 2 : same content line by line
        for i, (row_cont, row_seg) in enumerate(zip(reader_cont, reader_seg)):
            assert row_cont == row_seg, f"CSV files differ at line {i+1} : {row_cont} != {row_seg}"