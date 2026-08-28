use std::error::Error;
use std::env;
use std::fs;

use quick_xml::de;

use graph::digest_map;
use graph::Network;
use msccl_view::Algo;

fn count_tot_deg(topo: &Network) -> usize {
    topo.all_edges().fold(0usize, |acc, x| acc + *(x.2))
}

fn count_util(topo: &mut Network, sch: &Algo) {
	let orig_topo = topo.clone();
	let mut tot_use: usize = 0;
	for gpu in sch.gpus.iter() {
		for port in gpu.ports.iter() {
			if port.send == -1 { continue; }
			if let Some(e) = topo.edge_weight_mut(gpu.id, port.send as usize) {
				*e += port.steps.len();
				tot_use += port.steps.len();
			} else {
				panic!(
					"Edge {}->{} Doesn't Exist! Got the correct schedule network pair?",
					gpu.id,
					port.send,
				);
			}
		}
	}
	let mut max_t: usize = 0;
	for (src, dst, deg) in topo.all_edges() {
		let graph_deg = orig_topo.edge_weight(src, dst).unwrap();
		max_t = std::cmp::max(max_t, (*deg as f64 / *graph_deg as f64).ceil() as usize);
	}
	println!("Utilization {}", tot_use as f64 / (max_t * count_tot_deg(&orig_topo)) as f64 * 100.0);
}

fn main() -> Result<(), Box<dyn Error>> {
	let args: Vec<String> = env::args().collect();
	let fname = &args[1];
	let topo = &args[2];
	let mut network = digest_map(topo)?;
	let schedule_file = fs::read_to_string(fname)?;
	let schedule: Algo = de::from_str(&schedule_file)?;
	count_util(&mut network, &schedule);
	Ok(())
}
