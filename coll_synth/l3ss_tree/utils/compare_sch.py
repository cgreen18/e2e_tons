import subprocess
from pathlib import Path
import re
import csv
import sys
# import matplotlib.pyplot as plt

net_path = Path("./networks/tpu/")
# sch_path = Path("./schedule/")
fcoll_path = Path("../fcoll-sch/tpu_sch/sch")

def capture_output(out: str) -> list:
        p_info = re.compile(r"Network \w+\.map has (?P<ncount>\d+) nodes, (?P<ecount>\d+) edges")
        p_chunk = re.compile(r"(?P<chunks>\d+) chunks")
        p_time = re.compile(r"(?P<steps>\d*\.?\d+) steps per")
        p_waste = re.compile(r"Utilization (?P<waste>\d*\.?\d+)%")
        p_rtime = re.compile(r"Scheduling took (?P<time>\d*\.?\d+)s")
        util_match = p_waste.search(out)
        util = float(util_match.group("waste"))
        match_t = p_time.search(out)
        match_chunk = p_chunk.search(out)
        match_info = p_info.search(out)
        match_rtime = p_rtime.search(out)
        # return [match_info.group("ncount"), match_info.group("ecount"), match_chunk.group("chunks"), match_t.group("steps"), util, match_rtime.group("time")]
        return [match_chunk.group("chunks"), match_t.group("steps"), util, match_rtime.group("time")]

def append_csv(data: list):
        with open("data.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(data)

append_csv(["schedule", "V", "E", "chunks", "avg_steps", "utilization", "runtime"])

def parse_fcoll():
        for sch in fcoll_path.glob("*.xml"):
                print(f"Resolving for {sch}")
                result = subprocess.run(
                        ["./target/release/forestcoll_view", str(sch), str(net_path / sch.stem) + ".map"], 
                        capture_output=True,
                        text=True,
                )
                print(["./target/release/forestcoll_view", str(sch), str(net_path / sch.stem) + ".map"])
                print(result.stdout)
                append_csv([str(sch.stem)] + capture_output(result.stdout))

def parse_run():
        for network in net_path.glob("*.map"):
                print(f"Resolving for {network}")
                # run multi-tree multiple times due to randomness
                best_time = 1<<31;
                mult_result = subprocess.run(
                        ["./target/release/l3ss_tree", "build", str(network), "ag", "rand", "2"], 
                        capture_output=True,
                        text=True,
                )
                print(["./target/release/l3ss_tree", "build", str(network), "ag", "rand", "2"])
                print(mult_result.stdout)
                # # parse tacos schedule
                # tacos_result = subprocess.run(
                #       ["./target/release/l3ss_tree", "view", str(network), "--sch", str(sch_path / network.name) + ".xml"], 
                #       capture_output=True,
                #       text=True,
                # )
                curr_out = capture_output(mult_result.stdout)
                if float(curr_out[3]) < float(best_time):
                        best_time = curr_out[3]
                        best_out = curr_out
                append_csv([str(network.stem)] + best_out)

def scan_chunk():
        best_time = float(1<<31);
        chunks = list()
        time = list()
        for i in range(1, int(sys.argv[2])):
                mult_result = subprocess.run(
                                ["./target/release/l3ss_tree", "build", sys.argv[1], "ag", "rr", f"{i}"], 
                                capture_output=True,
                                text=True,
                        )
                # print(["./target/release/l3ss_tree", "build", sys.argv[1], "ag", "rr", f"{i}"])
                # print(mult_result.stdout)
                curr_out = capture_output(mult_result.stdout)
                chunks.append(float(curr_out[0]))
                time.append(float(curr_out[1]))
                if float(curr_out[1]) < float(best_time):
                        best_time = curr_out[1]
                        best_out = curr_out
        print(best_out)
        plt.scatter(chunks, time)
        plt.xlabel("chunks")
        plt.ylabel("time")
        plt.show()

if __name__ == "__main__":
        # parse_fcoll()
        # parse_run()
        scan_chunk()
