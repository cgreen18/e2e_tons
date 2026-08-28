use super::BuildContext;
use std::collections::HashSet;
use log::trace;
use petgraph::Direction;
use graph::{Network, ScheTree, ScheduleTrees};

const BIT16_MASK: usize = 0xffff;

/// Converts (switch, port) tuple to a single usize representing dummy node id.
/// Limit to 2^16 nodes in network and (usize_bits - 16) bit switch id.
pub(super) fn to_dummy_id(sid: usize, pid: usize) -> usize {
	sid << 16 | pid
}

/// Converts back from dummy id to tuple (switch, port) id.
pub(super) fn from_dummy_id(did: usize) -> (usize, usize) {
	(did >> 16, did & BIT16_MASK)
}

/// NOTE: sid needs to be >= 1 throughout this module, or everything explodes!!
pub(super) fn is_dummy_id(did: usize) -> bool {
	did > BIT16_MASK
}

/// Converts one switch to dummy-fc in graph
fn kill_switch(topo: &mut Network, sid: usize, slim: usize) {
	trace!("Killing switch {}", sid);
	// keep track of all dummy ids we added to FC them together
	let mut new_did: HashSet<usize> = HashSet::new();
	let mut new_edges: Vec<(usize, usize, usize)> = Vec::new();
	for (_, dst, cap) in topo.edges_directed(sid, Direction::Outgoing) {
		let dst_conv = if dst >= slim { to_dummy_id(dst, sid) } else { dst };
		let dummy = to_dummy_id(sid, dst);
		new_edges.push((dummy, dst_conv, *cap));
		trace!("Add edge ({:x}, {:x}) -> ({:x}, {:x}, {})", sid, dst, dummy, dst, *cap);
		new_did.insert(dummy);
	}
	for (src, _, cap) in topo.edges_directed(sid, Direction::Incoming) {
		let src_conv = if src >= slim { to_dummy_id(src, sid) } else { src };
		let dummy = to_dummy_id(sid, src);
		new_edges.push((src_conv, dummy, *cap));
		trace!("Add edge ({:x}, {:x}) -> ({:x}, {:x}, {})", src, sid, src_conv, dummy, *cap);
		new_did.insert(dummy);
	}
	// commit all edges
	for (src, dst, cap) in new_edges {
		topo.add_edge(src, dst, cap);
	}
	// Fully connect
	for src in new_did.iter() {
		for dst in new_did.iter() {
			if src != dst {
				topo.add_edge(*src, *dst, usize::MAX);
				topo.add_edge(*dst, *src, usize::MAX);
			}
		}
	}
}

/// Build a new BuildContext, with switches removed by replacing all ports in a switch
/// with a dummy node and fully connect all such dummy nodes in one switch with
/// infinite (usize::MAX) capacity edges.
pub(super) fn new<'a>(
	topo: &'a Network,
	slim: usize,
	nchunks: Option<usize>
) -> BuildContext<'a> {
	let mut count_topo = topo.clone();
	let (num_steps, mut num_chunks) = super::calc_opt_step(topo, slim);
	if let Some(mnchunks) = nchunks { num_chunks = mnchunks; }
	let _: Vec<_> = count_topo
		.all_edges_mut()
		.map(|x| { *(x.2) *= num_steps; })
		.collect();

	let mut schs: Vec<ScheduleTrees> = Vec::new();
	for _ in 0..num_chunks {
		let sch: ScheduleTrees = graph::new_sche(slim);
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
	completed.resize_with(num_chunks,|| vec![false; slim]);

	for sid in slim..topo.node_count() {
		kill_switch(&mut count_topo, sid, slim);
	}
	for sid in slim..topo.node_count() {
		count_topo.remove_node(sid);
	}

	BuildContext {
		topo_orig: topo,
		topo: count_topo,
		schs: schs,
		completed: completed,
		sid_start: slim,
		chunk_done: vec![false; num_chunks],
	}
}

/// Converts dummy node back to switch on one schedule tree.
fn revive_tree(tree: &mut ScheTree, slim: usize) {
	let mut commit_list: Vec<(usize, usize)> = Vec::new();
	let mut delete_list: Vec<(usize, usize)> = Vec::new();
	let mut node_delete: HashSet<usize> = HashSet::new();

	// first pass, remove leaf dummies
	for node in tree.nodes() {
		if is_dummy_id(node) && tree.neighbors(node).count() == 0 {
			node_delete.insert(node);
		}
	}
	for node in node_delete.drain() {
		tree.remove_node(node);
	}

	// second pass, conversion
	for (src, dst, _) in tree.all_edges_mut() {
		// both are dummy.
		// If they are from the same switch eliminate this edge.
		// If not, add back the inter switch connection.
		if is_dummy_id(src) && is_dummy_id(dst) {
			delete_list.push((src, dst));
			let (s_sid, _) = from_dummy_id(src);
			let (d_sid, _) = from_dummy_id(dst);
			if s_sid != d_sid { commit_list.push((s_sid, d_sid)); }
			node_delete.insert(src);
			node_delete.insert(dst);
		} else if is_dummy_id(src) || is_dummy_id(dst) {
			// one of them is dummy, essentially replace that node.
			delete_list.push((src, dst));
			let src_conv = if is_dummy_id(src) { 
				node_delete.insert(src);
				(from_dummy_id(src)).0 
			} else { src };
			let dst_conv = if is_dummy_id(dst) {
				node_delete.insert(dst);
				(from_dummy_id(dst)).0
			} else { dst };
			commit_list.push((src_conv, dst_conv));
		}
	}

	for (src, dst) in delete_list {
		tree.remove_edge(src, dst);
	}
	for (src, dst) in commit_list {
		// fix the diamond shape on switches
		if !tree.contains_node(dst) || dst < slim {
			tree.add_edge(src, dst, 0);
		}
	}
	for node in node_delete {
		tree.remove_node(node);
	}
}

/// Converts schedule back to original graph, converting dummy nodes back to switch nodes.
pub(super) fn revive_switch(sch: &mut [ScheduleTrees], slim: usize) {
	for chunk_sch in sch.iter_mut() {
		for tree_o in chunk_sch.iter_mut() {
			if let Some(tree) = tree_o.as_mut() {
				revive_tree(tree, slim);
			}
		}
	}
}
