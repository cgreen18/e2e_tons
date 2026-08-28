//! Crate for graph helper functions like digesting .map files
//! and algorithms.
pub use petgraph::graphmap::{DiGraphMap, UnGraphMap, GraphMap};

use std::collections::HashSet;
use std::error::Error;
use std::{fs, fs::OpenOptions};
use std::io::prelude::*;
use std::mem;
use std::str;

use layout::backends::svg::SVGWriter;
use layout::gv;
use log::{trace, info};
use petgraph::dot::Dot;
use petgraph::Direction;
use petgraph::EdgeType;

/// Marks node with number identifier, marks edge for used/not used in
/// schedule construction.
pub type Network = DiGraphMap<usize, usize>;

/// Single schedule tree.
///
/// Edges associated with time of addition.
pub type ScheTree = DiGraphMap<usize, usize>;

/// Forest containing schedule trees.
///
/// Tree rooted at node i sits at index i.
pub type ScheduleTrees = Vec<Option<ScheTree>>;

/// Generating graphgiz .dot representation of graph.
fn draw_graph(g: &Network) -> String {
	format!("{}", Dot::new(g))
}

/// Builds a new ScheduleTree containing all Nones.
pub fn new_sche(size: usize) -> ScheduleTrees {
	let mut sche: ScheduleTrees = Vec::new();
	sche.resize(size, None);
	sche
}

/// Reads a .map adjacency matrix file and return a petgraph::DiGraphMap.
///
/// Nodes are numbered from 0 to usize::MAX, according to order of row and
/// column in file. Currently all edges have 0usize associated to them for
/// weird printing issue of Dot.
pub fn digest_map(
	fname: &str,
) -> Result<Network, Box<dyn Error>> {
	let raw_file = fs::read(fname)?;
	let file: &str = std::str::from_utf8(&raw_file)?;
	// println!("{file}");

	let mut graph = Network::new();
	for (node, line) in file.lines().enumerate() {
		for (conn_node, is_edge) in line
			.split_ascii_whitespace()
			.enumerate() {
			if is_edge != "0" {
				graph.add_edge(
					node,
					conn_node,
					is_edge.parse::<usize>().expect("Illegal edge eegree in map!")
				);
			}
		}
	}
	Ok(graph)
}

pub fn count_tot_deg(topo: &Network) -> usize {
    topo.all_edges().fold(0usize, |acc, x| acc + *(x.2))
}

/// Returns an svg file given graph .dot file
fn draw_svg(dot: &str) -> String {
	let mut parser = gv::DotParser::new(dot);
	let mut vg_builder = gv::GraphBuilder::new();
	vg_builder.visit_graph(&parser.process().unwrap());
	let mut vg = vg_builder.get();
	let mut svg = SVGWriter::new();
	vg.do_it(false, false, false, &mut svg);
	svg.finalize()
}

/// Outputs the graph in svg file.
pub fn visualize_graph(g: &Network, fname: &str) -> Result<(), Box<dyn Error>> {
	// let svg_content = exec_dot(g.draw_graph(), vec![Format::Svg.into()]).unwrap();
	fs::write(fname, &draw_svg(&draw_graph(g)))?;
	Ok(())
}

/// Generates a .html file containing visualization of all schedule trees
pub fn visualize_schedule(
	s: &ScheduleTrees,
	fname: &str,
) -> Result<(), Box<dyn Error>> {
	let mut graph = String::new();
	fs::write(fname, "")?; // Clear the file first
	let mut f = OpenOptions::new()
		.append(true)
		.open(fname)?;
	for sche_opt in s.iter() {
		let Some(sche) = sche_opt else { continue; };
		let svg_content = draw_svg(&draw_graph(sche));
		graph.clear();
		graph.push_str("<div style=\"display: inline-block; width: 100%\">\n");
		graph.push_str(&svg_content);
		graph.push_str("</div>\n");
		write!(f, "{}", graph)?;
	}
	Ok(())
}

/// Graph algorithm module
pub mod algo {
	use std::iter;
	use super::*;
	/// Finds diameter of graph and gives path.
	/// 
	/// TODO: Use dynamic programming to simplify calculation.
	pub fn find_diameter(g: &Network) -> Vec<usize> {
		let mut dia: usize = 0;
		let mut depth_map: Vec<usize> = Vec::new();
		let mut last_node: usize = 0;
		for head in 0..g.node_count() {
			let (curr_dia, depth) = bfs(g, head);
			trace!("Node {}, Depth {}, depth marking {:?}", head, curr_dia, depth);
			if curr_dia > dia {
				for (i, node_depth) in depth.iter().enumerate() {
					if *node_depth == curr_dia {
						last_node = i;
						break;
					}
				}
				dia = curr_dia;
				depth_map = depth;
			}
		}
		// Backtrace from node to form path 
		let mut path: Vec<usize> = Vec::new();
		path.push(last_node);
		for step in (0..dia).rev() {
			for prev_node in g.neighbors(last_node) {
				if depth_map[prev_node] == step {
					path.push(prev_node);
					last_node = prev_node;
					break;
				}
			}
		}
		(&mut path).reverse();
		info!("Network has Diameter {}, path {:?}", dia, path);
		path
	}

	pub fn find_radix(g: &Network, dir: Direction) -> usize {
		let mut radix: usize = usize::MAX;
		for node in g.nodes() {
			let rdx_node = g
			    .edges_directed(node, dir)
			    .fold(0usize, |acc, x| acc + *(x.2));
			if radix > rdx_node {
				radix = rdx_node;
			}
		}
		radix
	}

	/// Holds iterator in which calling next() yields the next bfs layer.
	///
	/// Note return of BfsIterator is owned. Ignores any meaning of E.
	pub struct BfsIterator<'a, E, Ty> {
		g: &'a GraphMap<usize, E, Ty>,
		head: usize,
		last_leaves: HashSet<usize>,
		next_leaves: HashSet<usize>,
		seen_node: usize,
		dist: Vec<usize>,
		time: usize,
	}

	impl<'a, E, Ty> Iterator for BfsIterator<'a, E, Ty>
	where
		Ty: EdgeType,
	{
		type Item = Vec<usize>;

		fn next(&mut self) -> Option<Self::Item> {
			if self.seen_node == self.g.node_count() {
				return None;
			}
			let mut explored: Vec<usize> = Vec::new();
			self.time += 1;
			for start_node in self.last_leaves.iter() {
				for (_, next_node, _) in 
					self.g.edges(*start_node) {
					if next_node != self.head && self.dist[next_node] == 0 {
						self.next_leaves.insert(next_node);
						explored.push(next_node);
						self.dist[next_node] = self.time;
						self.seen_node += 1;
					}
				}
			}
			mem::swap(&mut self.last_leaves, &mut self.next_leaves);
			self.next_leaves.clear();
			Some(explored)
		}
	}

	pub fn bfs_iter<E, Ty>(g: &GraphMap<usize, E, Ty>, head: usize) -> BfsIterator<'_, E, Ty>
	where
		Ty: EdgeType,
	{
		BfsIterator {
			dist: vec![0; g.node_count()],
			g,
			head,
			last_leaves: HashSet::from([head]),
			next_leaves: HashSet::new(),
			seen_node: 1,
			time: 0,
		}
	}

	/// Perform BFS on network starting from head.
	///
	/// Returns max depth and nodes mark with depth.
	pub fn bfs(g: &Network, head: usize) -> (usize, Vec<usize>) {
		let mut iterator = bfs_iter(g, head);
		while let Some(_) = iterator.next() { };
		(iterator.time, iterator.dist)
	}
}
