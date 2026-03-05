import json
from pathlib import Path


def get_memory_samples_from_json(path: str):

    with open(path) as f:
        data = json.load(f)

    events = data["traceEvents"]
    # get only [memory] events
    mem_events = filter(lambda events: events["name"] == "[memory]", events)
    # extract memory allocated in each event
    memory = [mem["args"]["Total Allocated"] for mem in mem_events]

    return memory


# Display memory peak
def display_memory_peak(log_dir, verbose=False):
    jsons_generated = list(Path(log_dir).glob("*.json"))
    for i, jj in enumerate(jsons_generated):
        memory = get_memory_samples_from_json(jj)
        peak = max(memory)
        print(f"Repeat #{i}:{jj} --> memory peak: {peak/(1024**3):.2f} GB")
        if verbose:
            print(f"Raw data of memory foorprint: {memory}")
    # return the last measured peak
    return peak, memory
