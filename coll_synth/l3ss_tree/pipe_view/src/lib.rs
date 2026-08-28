use std::fs;
use std::error::Error;

use graph::{ScheTree, ScheduleTrees};

use serde::Deserialize;
use quick_xml::de;

#[derive(Debug, Deserialize, PartialEq)]
struct Algo {
	#[serde(rename = "@nchunkspernode")]
	nchunks: usize,
	#[serde(default, rename = "tree")]
	trees: Vec<Tree>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Tree {
	#[serde(rename = "@root")]
	root: usize,
	#[serde(rename = "@index")]
	index: usize,
	#[serde(rename = "@nchunks")]
	nchunks: usize,
	#[serde(rename = "@height")]
	height: usize,
	#[serde(default, rename = "send")]
	sends: Vec<Comms>
}

#[derive(Debug, Deserialize, PartialEq)]
struct Comms {
	#[serde(rename = "@src")]
	src: usize,
	#[serde(rename = "@dst")]
	dst: usize,
	#[serde(rename = "@path")]
	path: String,
}

fn count_trees(algo: &Algo) -> usize {
	algo.trees.iter().map(|t| t.nchunks).sum()
}

/// Finds a chunk schedule for which tree rid is not populated yet, return chunk id.
fn find_empty(schedule: &[ScheduleTrees], rid: usize) -> usize {
	for (i, chunk) in schedule.iter().enumerate() {
		if chunk[rid]
			.as_ref()
			.expect(&format!("id {} out of bound!", rid))
			.node_count() 
			== 0 { 
			return i; 
		}
	}
	panic!("Did not find a place for tree {:?}, check nchunkspernode!", rid);
}

/// Builds tree described by schedule into tree.
fn build_tree(schedule: &Tree, tree: &mut ScheTree) {
	tree.add_node(schedule.root);
	for edge in schedule.sends.iter() {
		tree.add_edge(edge.src, edge.dst, 0);
	}
}

/// Builds internal data structure from a pipelined tree description.
///
/// Note this function expects standard node naming expectations of emap files.
pub fn build(fname: &str) -> Result<Vec<ScheduleTrees>, Box<dyn Error>> {
	let schedule_file = fs::read_to_string(fname)?;
	let mut algo: Algo = de::from_str(&schedule_file)?;
	let num_trees = count_trees(&algo) / algo.nchunks;
	let mut schedule: Vec<ScheduleTrees> = vec![vec![Some(ScheTree::new()); num_trees]; algo.nchunks];
	for tree in algo.trees.iter_mut() {
		while tree.nchunks > 0 {
			let chunk_loc = find_empty(&schedule, tree.root);
			build_tree(tree, schedule[chunk_loc][tree.root].as_mut().unwrap());
			tree.nchunks -= 1;
		}
	}
	Ok(schedule)
}
