pub use graph::{Network, ScheTree, ScheduleTrees, algo};

use std::cmp::min;
use std::collections::{HashSet, VecDeque};

use log::{debug, info};
use petgraph::Direction;

/// Checks time dependency. For each node:
///
/// max(incoming_time) < min(outgoing_time)
fn check_tree_time(sche: &ScheTree) {
	for node in sche.nodes() {
		let incoming_max: Option<usize> = sche
			.edges_directed(node, Direction::Incoming)
			.map(|i| *(i.2))
			.max();
		let outgoing_min: Option<usize> = sche
			.edges_directed(node, Direction::Outgoing)
			.map(|i| *(i.2))
			.min();
		let Some(i_max) = incoming_max else { continue; };
		let Some(o_min) = outgoing_min else { continue; };
		debug!("Node {}, i_max {}, o_min {}", node, i_max, o_min);
		assert!(i_max < o_min);
	}
}

/// Checks a schedule is legal. Panics with assert! if any problems discovered.
fn check_schedule(
	sche_chunks: &[ScheduleTrees],
	network: &Network,
	time: usize,
	is_pipe: bool,
) {
	info!("Check: tree complete?");
	for chunk in sche_chunks.iter() {
		for tree_o in chunk.iter() {
			if let Some(tree) = tree_o {
				assert!(tree.node_count() == network.node_count());
			}
		}
	}
	if is_pipe { return; }
	info!("Check: Chunk dependency in time");
	for (c, sche) in sche_chunks.iter().enumerate() {
		info!("Checking chunk {}", c);
		for (i, sch_o) in sche.iter().enumerate() {
			let Some(sch) = sch_o else { continue; };
			info!("Check tree {}", i);
			// Check: Time marking should be monotonic
			check_tree_time(&sch);

			// Check: Should be exactly n-1 edges, contains all nodes
			assert!(sch.edge_count() == network.node_count() - 1);
			let mut node_set: HashSet<usize> = HashSet::new();
			node_set.extend(sch.nodes().collect::<Vec<_>>().iter());
			// exploits our numbering scheme
			assert!(node_set.len() == network.node_count());
			for node in node_set.iter() {
				assert!(*node < node_set.len());
			}
		}
	}
	// Check: No edge double taking in each time step
	// Assumes network is clean (i.e. no edges marked)
	info!("Check: double taking");
	let mut time_topo: VecDeque<Network> = VecDeque::new();
	// can't just store time copies of network, some buffering
	let mut queue_start: usize = 0;
	let buf_step: usize = (time as f64 / sche_chunks.len() as f64).ceil() as usize;
	for sche in sche_chunks {
		// flush in all the schedule first
		// allocate
		time_topo.resize_with(
			min(time - queue_start, buf_step) + time_topo.len(),
			|| network.clone(),
		);
		// Go in and take edges
		for sch_o in sche.iter() {
			let Some(sch) = sch_o else { continue; };
			for (from_node, to_node, time) in sch.all_edges() {
				assert!(*time >= queue_start);
				if *time - queue_start >= time_topo.len() {
					// some networks not filled at end is fine
					time_topo.resize_with(
						time_topo.len() + *time - queue_start,
						|| network.clone(),
					);
				}
				let taken = time_topo[*time - queue_start]
					.edge_weight_mut(from_node, to_node)
					.unwrap();
				assert!(*taken != 0);
				*taken -= 1;
			}
		}
		// retire networks
		loop {
			let Some(queue_front) = time_topo.front() else { break; };
			let mut edges_taken = true;
			for (_, _, taken) in queue_front.all_edges() {
				edges_taken = edges_taken && *taken == 0;
			}
			if edges_taken {
				time_topo.pop_front();
				queue_start += 1;
			} else {
				break;
			}
		}
	}
}

fn count_max_step(topo: &Network, sche_trees: &[ScheduleTrees]) -> usize {
	let mut count_network = topo.clone();
	for (_, _, cap) in count_network.all_edges_mut() {
		*cap = 0;
	}
	for chunk_sch in sche_trees.iter() {
		for tree_o in chunk_sch.iter() {
			if let Some(tree) = tree_o.as_ref() {
				for (src, dst, _) in tree.all_edges() {
					*(count_network
						.edge_weight_mut(src, dst)
						.expect(&format!("Missing edge ({}, {}) in network!", src, dst))
					) += 1;
				}
			}
		}
	}
	let mut max_step: usize = 0;
	for (src, dst, cap) in count_network.all_edges() {
		let curr_step = (*cap as f64 / *topo.edge_weight(src, dst).unwrap() as f64)
			.ceil() as usize;
		if curr_step > max_step { max_step = curr_step; }
	}
	max_step
}

/// Prints metrics of schedule to stdout.
pub fn stat(
	sche_trees: &[ScheduleTrees],
	network: &Network,
	no_check: bool,
	is_pipe: bool,
) {
	// general stat
	let max_step: usize = count_max_step(network, sche_trees);
	// edge wastage
	let mut use_edge: usize = 0;

	for chunk_sch in sche_trees {
		for (_, sch_o) in chunk_sch.iter().enumerate() {
			if let Some(sch) = sch_o.as_ref() {
				use_edge += sch.edge_count();
			}
		}
	}
	
	if !no_check {
		check_schedule(sche_trees, network, max_step, is_pipe);
	}

	println!(
		"{} * {} trees. {} steps. {} steps per chunk",
		sche_trees.len(), network.node_count(),
		max_step,
		max_step as f64 / sche_trees.len() as f64
	);
	let avg_use: f64 = use_edge as f64 / max_step as f64;
	let deg_sum = network.all_edges().fold(0usize, |acc, e| { acc + *(e.2) }) as f64;
	let util = avg_use / deg_sum;
	println!("Utilization {:.4}%", util * 100.0f64);
}
