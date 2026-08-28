/// Build disjoint spanning trees for pipelined schedule
/// with the co-bfs algorithm.
pub mod killswitch;

pub use graph::{Network, ScheTree, ScheduleTrees};

use log::{info, trace};
use num::integer::gcd;
use petgraph::Direction;
use rand;
use rand::seq::SliceRandom;

struct BuildContext<'a> {
	topo_orig: &'a Network,
	topo: Network,
	schs: Vec<ScheduleTrees>,
	completed: Vec<Vec<bool>>,
	chunk_done: Vec<bool>,
	sid_start: usize,
}

impl<'a> BuildContext<'a> {
	fn new (
		topo: &Network,
		slim: Option<usize>,
		nchunks: Option<usize>,
	) -> BuildContext<'_> {
		let mut count_topo = topo.clone();
		let node_cnt = slim.unwrap_or_else(|| topo.node_count());
		let (num_steps, mut num_chunks) = calc_opt_step(topo, node_cnt);
		info!("{} chunks, {} steps to reach 100% utilization.", num_chunks, num_steps);
		if let Some(mnchunks) = nchunks { num_chunks = mnchunks; }
		let _: Vec<_> = count_topo.all_edges_mut().map(|x| { *(x.2) *= num_steps; () }).collect();

		let mut schs: Vec<ScheduleTrees> = Vec::new();
		for _ in 0..num_chunks {
			let sch: ScheduleTrees = graph::new_sche(node_cnt);
			let sch: ScheduleTrees = sch
				.into_iter()
				.enumerate()
				.map(|(i, _x)| {
					let mut tree = ScheTree::new();
					tree.add_node(i);
					Some(tree)
				})
				.collect();
			schs.push(sch);
		}

		let mut completed: Vec<Vec<bool>> = Vec::new();
		completed.resize_with(num_chunks,|| vec![false; node_cnt]);

		BuildContext {
			sid_start: node_cnt,
			topo_orig: topo,
			topo: count_topo,
			schs: schs,
			completed: completed,
			chunk_done: vec![false; num_chunks],
		}
	}
}

/// Gives LCM(1, (N-1)/min_cut) as min number of chunks
/// Returns: (num_steps, num_chunks)
fn calc_opt_step(topo: &Network, slim: usize) -> (usize, usize) {
	let min_cut = graph::algo::find_radix(topo, Direction::Incoming);
	let num_chunks = min_cut / gcd(slim - 1, min_cut);
	let num_step = num_chunks * (slim - 1) / min_cut;
	(num_step, num_chunks)
}

fn tree_add_edge(topo: &mut Network, tree: &mut ScheTree) -> bool {
	let mut rng = rand::rng();
	let mut from_list: Vec<_> = tree
		.nodes()
		.collect();
	from_list.shuffle(&mut rng);
	for src in from_list {
		let mut to_list: Vec<_> = topo
			.neighbors_directed(src, Direction::Outgoing)
			.collect();
		to_list.shuffle(&mut rng);
		for dst in to_list {
			let capacity = topo.edge_weight_mut(src, dst).unwrap();
			if !tree.contains_node(dst) && *capacity > 0 {
				tree.add_edge(src, dst, 0);
				*capacity -= 1;
				trace!("Adding edge ({:x}, {:x})", src, dst);
				return true;
			}
		}
	}
	return false;
}

fn cobfs_pass(ctx: &mut BuildContext) -> bool {
	let mut forward_progress: bool = false;
	for (chunk, sch) in ctx.schs.iter_mut().enumerate() {
		trace!("Chunk {}", chunk);
		for (tid, tree_o) in sch.iter_mut().enumerate() {
			if !ctx.completed[chunk][tid] && let Some(tree) = tree_o.as_mut() {
				trace!("tree {}", tid);
				forward_progress |= tree_add_edge(&mut ctx.topo, tree);
			}
		}
	}
	forward_progress
}

/// Get you one more set of edges aka step to work with.
///
/// Deals with switched converted gracefully.
fn add_step(ctx: &mut BuildContext) {
	let (num_step, _) = calc_opt_step(&ctx.topo_orig, ctx.sid_start);
	for (src, dst, orig_cap) in ctx.topo_orig.all_edges() {
		let mut src_conv = if src >= ctx.sid_start {
			killswitch::to_dummy_id(src, dst)
		} else { src };
		if !ctx.topo.contains_node(src_conv) { src_conv = src; };
		let mut dst_conv = if dst >= ctx.sid_start {
			killswitch::to_dummy_id(dst, src)
		} else { dst };
		if !ctx.topo.contains_node(dst_conv) { dst_conv = dst; };
		trace!("Adding step for ({:x}, {:x}) -> ({:x}, {:x})", src, dst, src_conv, dst_conv);
		*(ctx.topo.edge_weight_mut(src_conv, dst_conv).unwrap()) += *orig_cap * num_step;
	}
}

/// Counts if tree is done. If so, then mark.
/// Also count the number of chunks newly done and returns it.
fn mark_complete(ctx: &mut BuildContext) -> usize {
	trace!("Build stat:");
	let mut num_complete: usize = 0;
	for chunk in 0..ctx.schs.len() {
		if ctx.chunk_done[chunk] { continue; }
		let mut chunk_done: bool = true;
		for i in 0..ctx.schs[chunk].len() {
			let tree = ctx.schs[chunk][i].as_ref().unwrap();
			// let mut tree_done: bool = tree.node_count() >= ctx.topo.node_count();
			let mut tree_done: bool = true;
			for nid in ctx.topo.nodes() {
				if !killswitch::is_dummy_id(nid) && !tree.contains_node(nid) {
					tree_done = false;
					break;
				}
			}
			ctx.completed[chunk][i] = tree_done;
			chunk_done &= tree_done;
			trace!(
				"tree [{}][{}]: {}/{} nodes {:?}, done {}",
				chunk,
				i,
				tree.node_count(),
				ctx.topo.node_count(),
				tree.nodes().collect::<Vec<_>>(),
				tree_done,
			);
		}
		if chunk_done { 
			ctx.chunk_done[chunk] = true;
			num_complete += 1;
		}
	}
	num_complete
}

const EXP_THRESH: usize = 1;

pub fn build (
	topology: &Network, 
	switch: Option<usize>,
	use_legacy: bool,
	nchunks: Option<usize>,
) -> Vec<ScheduleTrees> {
	let mut ctx: BuildContext = if use_legacy && let Some(sid) = switch {
		killswitch::new(topology, sid, nchunks)
	} else {
		BuildContext::new(&topology, switch, nchunks)
	};
	let mut num_complete: usize = 0;
	let mut tried_expand: usize = 0;
	while num_complete < ctx.schs.len() {
		trace!("Iteration!");
		trace!("Tried expanding {} times.", tried_expand);
		let forward_progress = cobfs_pass(&mut ctx);
		if forward_progress { tried_expand = 0; }
		num_complete += mark_complete(&mut ctx);

		// Try adding one more copy of edge to see if it moves. If not, problem.
		if !forward_progress && num_complete < ctx.schs.len() {
			if tried_expand > EXP_THRESH { panic!("Deadlock!!"); }
			trace!("Adding 1 step!");
			add_step(&mut ctx);
			tried_expand += 1;
		}
		trace!("{} chunks done!", {num_complete});
	}
	if matches!(switch, Some(..)) && use_legacy {
		killswitch::revive_switch(&mut ctx.schs, ctx.sid_start);
	}
	ctx.schs
}

