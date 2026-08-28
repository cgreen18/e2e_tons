//! This lib converts schedule trees to MSCCL++ Python DSL file.
use std::collections::HashSet;
use std::error::Error;
use std::io::Write;
use std::fs::File;

use graph::algo;
use graph::{ScheduleTrees, ScheTree};

use log::trace;
use petgraph::Direction;

struct AlgoContext {
	channels: HashSet<(usize, usize)>,
	// each node each epoch
	dsl_buf: Vec<Vec<String>>,
	// for each chunk, each schedule, each layer
	pipe_layers: Vec<Vec<Vec<Vec<(usize, usize)>>>>,
	curr_packet: Vec<Vec<Vec<usize>>>,
	// where are we in terms of filling the pipeline?
	fill_head: Vec<Vec<usize>>,
	subdivide: usize,
}

impl AlgoContext {
	fn new(nchunk: usize, ntree: usize, subdivide: usize) -> AlgoContext {
		AlgoContext {
			channels: HashSet::new(),
			dsl_buf: vec![vec![String::new()]; ntree],
			pipe_layers: vec![vec![Vec::new(); ntree]; nchunk],
			curr_packet: vec![vec![Vec::new(); ntree]; nchunk],
			fill_head: vec![vec![1; ntree]; nchunk],
			subdivide: subdivide,
		}
	}
	fn advance_packet(self: &mut AlgoContext) {
		for (cid, chunk) in self.curr_packet.iter_mut().enumerate() {
			for (tid, tree) in chunk.iter_mut().enumerate() {
				trace!("Advance [{}][{}], head {}", cid, tid, self.fill_head[cid][tid]);
				for i in 0..self.fill_head[cid][tid] { tree[i] += 1; }
				trace!("pipeline {:?}", tree);
				if self.fill_head[cid][tid] < tree.len() { self.fill_head[cid][tid] += 1; }
			}
		}
		for buf in self.dsl_buf.iter_mut() {
			buf.push(String::new());
		}
	}
	fn init_chan(self: &mut AlgoContext, algo: &[ScheduleTrees]) {
		for chunk in algo.iter() {
			for tree_o in chunk.iter() {
				if let Some(tree) = tree_o {
					for (src, dst, _) in tree.all_edges() {
						self.channels.insert((src, dst));
					}
				}
			}
		}
	}
}

fn get_rank_name(gpuid: usize) -> String {
	format!("rank{}", gpuid)
}

fn get_buf_name(gpuid: usize) -> String {
	format!("buf{}", gpuid)
}

fn get_chan_name(from: usize, to: usize) -> String {
	format!("chan{}_{}", from, to)
}

fn ranks_buf_preamble<'a, N, C>(nodes: N, chans: C, chunking: usize) -> String 
where
	N: Iterator<Item = usize>,
	C: Iterator<Item = &'a (usize, usize)>,
{
	let mut preamble: String = String::new();
	for n in nodes {
		let my_idx: usize = chunking * n;
		preamble.push_str(&format!(
			"\t\t{} = rank.Rank({})\n",
			get_rank_name(n),
			n,
		));
		preamble.push_str(&format!(
			"\t\t{} = {}.get_output_buffer()\n",
			get_buf_name(n),
			get_rank_name(n),
		));
		preamble.push_str(&format!(
			"\t\t{}.copy({}[{}:{}], {}.get_input_buffer()[0:{}], tb=0)\n",
			get_rank_name(n),
			get_buf_name(n),
			my_idx,
			my_idx + chunking,
			get_rank_name(n),
			chunking,
		));
	}
	for (src, dst) in chans {
		preamble.push_str(&format!(
			"\t\t{} = channel.MemoryChannel({}, {})\n",
			get_chan_name(*src, *dst),
			dst,
			src,
		));
	}
	preamble
}

fn get_preemble(algo: &[ScheduleTrees], subdivide: usize) -> String {
	let mut out_dsl: String = String::new();
	// better mechanism coming
	let coll = "allgather";
	out_dsl.push_str("from mscclpp.language import *\n");
	out_dsl.push_str("def comm_program(spec):\n");
	out_dsl.push_str(&format!("\tnum_gpus = {}\n\tchunk_factor = {}\n", algo[0].len(), algo.len() * subdivide));
	out_dsl.push_str(&format!(
			"\tcollective = collectives.{}(num_gpus, chunk_factor, inplace=True)\n",
			if coll == "allreduce" { "AllReduce" } else { "AllGather" },
		));
	out_dsl.push_str("\twith program.CollectiveProgram(");
	out_dsl.push_str(&format!("\"l3ss_tree\","));
	out_dsl.push_str(&format!(
			"collective,num_gpus,protocol=\"Simple\",min_message_size=1,max_message_size=80<<30) as prog:\n",
		));
	out_dsl
}

fn collapse_switch(tree: &mut ScheTree, slim: usize) {
	for node in tree.nodes().collect::<Vec<_>>().into_iter() {
		if node >= slim {
			trace!("Dropping switch {}", node);
			let mut child_list: Vec<usize> = Vec::new();
			for (_, child, _) in tree.edges_directed(node, Direction::Outgoing) {
				child_list.push(child);
			}
			// should be a tree, so only 1 incoming
			// This needs to be patched for reduce schedule
			let (parent, _, _) = tree.edges_directed(node, Direction::Incoming).next().unwrap();
			for child in child_list {
				trace!("{} -> {} convert {} -> {}", node, child, parent, child);
				tree.remove_edge(node, child);
				tree.add_edge(parent, child, 0);
			}
			tree.remove_node(node);
		}
	}
}

fn init_layer(sch: &ScheTree, rid: usize) -> Vec<Vec<(usize, usize)>> {
	let mut layer_edges: Vec<Vec<(usize, usize)>> = Vec::new();
	let mut last_node: Vec<usize> = vec![rid];
	for dst_nodes in algo::bfs_iter(sch, rid) {
		let mut layer: Vec<(usize, usize)> = Vec::new();
		// find edge correspond to one child node
		for dst in dst_nodes.iter() {
			for src in last_node.iter() {
				if sch.contains_edge(*src, *dst) {
					layer.push((*src, *dst));
					break;
				}
			}
		}
		last_node = dst_nodes;
		layer_edges.push(layer);
	}
	layer_edges
}

/// writes schedule for all tree based on currenet ctx markings.
fn schedule_layer(ctx: &mut AlgoContext) {
	for (cid, chunk) in ctx.pipe_layers.iter().enumerate() {
		for (tid, pipe) in chunk.iter().enumerate() {
			trace!("chunk {}, tree {}", cid, tid);
			for eid in 0..ctx.fill_head[cid][tid] {
				if ctx.curr_packet[cid][tid][eid] < ctx.subdivide {
					for edge in pipe[eid].iter() {
						let src = edge.0;
						let dst = edge.1;
						let chunk_idx = 
							tid * ctx.pipe_layers.len() * ctx.subdivide
							+ cid * ctx.subdivide
							+ ctx.curr_packet[cid][tid][eid];
						trace!(
							"layer {}, edge {} -> {} subpacket {}",
							eid,
							src,
							dst,
							ctx.curr_packet[cid][tid][eid],
						);
						ctx.dsl_buf[src].last_mut().unwrap().push_str(&format!(
							"\t\t{}.put({}[{}:{}], {}[{}:{}], tb=0)\n",
							get_chan_name(src, dst),
							get_buf_name(dst),
							chunk_idx,
							chunk_idx + 1,
							get_buf_name(src),
							chunk_idx,
							chunk_idx + 1,
						));
					}
				}
			}
		}
	}
}

fn gen_sync(chan: &HashSet<(usize, usize)>, tid: usize) -> String {
	let mut sync_inst: String = String::new();
	let sync_chan: Vec<String> = chan
		.iter()
		.filter(|e| (*e).0 == tid)
		.map(|e| get_chan_name((*e).0, (*e).1))
		.collect();
	for chan in sync_chan.iter() {
		sync_inst.push_str(&format!(
			"\t\t{}.signal(tb=0, data_sync=internal.types.SyncType.before)\n",
			chan,
		));
	}
	for chan in sync_chan.iter() {
		sync_inst.push_str(&format!(
			"\t\t{}.wait(tb=0, data_sync=internal.types.SyncType.after)\n",
			chan,
		));
	}
	sync_inst
}

/// Writes actual pipelined schedule.
fn schedule_pipe(
	algo: &[ScheduleTrees],
	subdivide: usize,
	file: &mut File,
) -> Result<(), Box<dyn Error>> {
	let mut ctx = AlgoContext::new(algo.len(), algo[0].len(), subdivide);
	// init buffers, channels
	ctx.init_chan(algo);
	file.write(ranks_buf_preamble(
		algo[0][0].as_ref().unwrap().nodes(),
		ctx.channels.iter(),
		algo.len() * subdivide,
	).as_bytes())?;
	// init layers
	for (cid, chunk) in algo.iter().enumerate() {
		for (tid, tree_o) in chunk.iter().enumerate() {
			if let Some(tree) = tree_o.as_ref() {
				trace!("Discovering layer for {} {}", cid, tid);
				trace!("{}", tree.node_count());
				let new_layer = init_layer(tree, tid);
				ctx.curr_packet[cid][tid] = vec![0; new_layer.len()];
				ctx.pipe_layers[cid][tid] = new_layer;
			}
		}
	}
	// acutally schedule
	let mut done: bool = false;
	while !done {
		trace!("Epoch!");
		done = true;
		schedule_layer(&mut ctx);
		for chunk in ctx.curr_packet.iter() {
			for pipe in chunk.iter() {
				done &= *pipe.last().unwrap() >= ctx.subdivide - 1;
			}
		}
		if !done {
			ctx.advance_packet();
		}
	}
	trace!("Done scheduling, writing file...");
	// flush all strings
	for (tid, tree) in algo[0].iter().enumerate() {
		if let Some(_) = tree {
			file.write(&format!("\t\t#---------- GPU {} ----------\n", tid).as_bytes())?;
			file.write(gen_sync(&ctx.channels, tid).as_bytes())?;
			for epoch in ctx.dsl_buf[tid].iter() {
				file.write(epoch.as_bytes())?;
				file.write(gen_sync(&ctx.channels, tid).as_bytes())?;
			}
		}
	}
	Ok(())
}

/// Write MSCCL++ DSL to file, according to a pipelined schedule.
pub fn write_pipe(
	schedule: &[ScheduleTrees],
	slim: usize,
	subdivide: usize,
	fname: &str,
) -> Result<(), Box<dyn Error>> {
	let mut algo = Vec::from(schedule);
	for chunk in algo.iter_mut() {
		for tree in chunk.iter_mut() {
			if let Some(tree_i) = tree.as_mut() { collapse_switch(tree_i, slim); }
		}
	}
	let mut file = File::create(fname)?;
	file.write(get_preemble(&algo, subdivide).as_bytes())?;
	schedule_pipe(&algo, subdivide, &mut file)?;
	file.write("\t\treturn prog".as_bytes())?;
	Ok(())
}

pub fn write_step(schedule: &[ScheduleTrees], fname: &str) -> Result<(), Box<dyn Error>> {
	unimplemented!();
}
