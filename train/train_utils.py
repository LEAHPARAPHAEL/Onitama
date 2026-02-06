

def extract_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1
    
def extract_shard_idx(str : str):
    try:
        idx = int(str.split("_")[1])
        return idx
    except:
        return -1
    
def extract_model_steps(str : str):
    try:
        idx = int(str.split("_")[-1].split(".")[0])
        return idx
    except:
        return 0
    
def extract_positions(str : str):
    try:
        idx = int(str.split("_")[-1].split(".")[0])
        return idx
    except:
        return 0

