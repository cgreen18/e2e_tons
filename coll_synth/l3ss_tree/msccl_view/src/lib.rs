pub use graph::{Network, ScheTree, ScheduleTrees, algo};

use std::fs;
use std::error::Error;

use petgraph::Direction;
use serde::{Deserialize, Serialize};
use quick_xml::{se, de};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collective {
    AllGather,
    AllReduce,
    ReduceScatter,
}

impl Collective {
    fn msccl_name(self) -> &'static str {
        match self {
            Self::AllGather => "allgather",
            Self::AllReduce => "allreduce",
            Self::ReduceScatter => "reduce_scatter",
        }
    }

    fn phase_chunk_count(self, schedule_count: usize) -> usize {
        match self {
            Self::AllReduce => {
                assert!(schedule_count % 2 == 0, "AllReduce requires equal reduction and broadcast phases");
                schedule_count / 2
            },
            Self::AllGather | Self::ReduceScatter => schedule_count,
        }
    }

    fn is_reduction_schedule(self, schedule_index: usize, logical_chunks: usize) -> bool {
        match self {
            Self::AllGather => false,
            Self::ReduceScatter => true,
            Self::AllReduce => schedule_index < logical_chunks,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename = "algo")]
pub struct Algo {
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@proto")]
    proto: String,
    #[serde(rename = "@nchannels")]
    nchannels: usize,
    #[serde(rename = "@nchunksperloop")]
    nchunksperloop: usize,
    #[serde(rename = "@ngpus")]
    ngpus: usize,
    #[serde(rename = "@coll")]
    coll: String,
    /// Number of logical chunks in the per-rank workload input. This is an
    /// extension consumed by the Chakra lowering path; MSCCL readers ignore it.
    #[serde(default, rename = "@input_chunks", skip_serializing_if = "Option::is_none")]
    input_chunks: Option<usize>,
    /// l3ss offsets are laid out as (subchunk * ranks + owning_rank).
    #[serde(default, rename = "@chunk_layout", skip_serializing_if = "Option::is_none")]
    chunk_layout: Option<String>,
    #[serde(rename = "@inplace")]
    inplace: u32,
    #[serde(rename = "@outofplace")]
    outofplace: u32,
    #[serde(rename = "@minBytes")]
    min_bytes: u32,
    #[serde(rename = "@maxBytes")]
    max_bytes: u32,
    #[serde(default, rename = "gpu")]
    pub gpus: Vec<Gpu>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Gpu {
    #[serde(rename = "@id")]
    pub id: usize,
    #[serde(rename = "@i_chunks")]
    i_chunks: usize,
    #[serde(rename = "@o_chunks")]
    o_chunks: usize,
    #[serde(rename = "@s_chunks")]
    s_chunks: usize,
    #[serde(default, rename = "tb")]
    pub ports: Vec<Port>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Port {
    #[serde(rename = "@id")]
    id: usize,
    #[serde(rename = "@send")]
    pub send: isize,
    #[serde(rename = "@recv")]
    recv: isize,
    #[serde(rename = "@chan")]
    chan: usize,
    #[serde(default, rename = "step")]
    pub steps: Vec<Step>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Step {
    #[serde(rename = "@s")]
    s: usize,
    #[serde(rename = "@type")]
    transact_type: String,
    #[serde(rename = "@srcbuf")]
    srcbuf: String,
    #[serde(rename = "@srcoff")]
    srcoff: usize,
    #[serde(rename = "@dstbuf")]
    dstbuf: String,
    #[serde(rename = "@dstoff")]
    dstoff: usize,
    #[serde(rename = "@cnt")]
    cnt: usize,
    #[serde(rename = "@depid")]
    depid: isize,
    #[serde(rename = "@deps")]
    deps: isize,
    #[serde(rename = "@hasdep")]
    hasdep: u32,
    /// Stable index in the invoking workload's per-rank input partition.
    #[serde(default, rename = "@logical_chunk", skip_serializing_if = "Option::is_none")]
    logical_chunk: Option<usize>,
}

pub fn build(fname: &str) -> Result<Vec<ScheduleTrees>, Box<dyn Error>> {
    let schedule_file = fs::read_to_string(fname)?;
    let schedule: Algo = de::from_str(&schedule_file)?;
    let num_nodes: usize = schedule.gpus.len();
    let num_chunks: usize = schedule.nchunksperloop / num_nodes;
    let mut sche_trees: Vec<ScheduleTrees> = Vec::new();
    let mut captured_edges: usize = 0;
    sche_trees.resize_with(
        num_chunks,
        || {
            let mut tree = ScheduleTrees::new();
            tree.resize_with(num_nodes, || Some(ScheTree::new()));
            tree
        },
    );
    for sche in sche_trees.iter_mut() {
        for (i, tree) in sche.iter_mut().enumerate() {
            tree.as_mut().unwrap().add_node(i);
        }
    }

    for gpu in schedule.gpus {
        let from_node = gpu.id;
        for port in gpu.ports {
            // only look at broadcast schedule, completely mirror of recv
            if port.send >= 0 {
                let to_node: usize = port.send as usize;
                for step in port.steps {
                    let chunk_step = step.srcoff / num_nodes;
                    let chunk_node = step.srcoff % num_nodes;
                    let Some(ref mut tree) = sche_trees[chunk_step][chunk_node] 
                        else { panic!("Tree {}-{} doesn't exist!", chunk_step, chunk_node); };
                    assert!(!tree.contains_edge(from_node, to_node), "Duplicating edge in schedule!");
                    tree.add_edge(from_node, to_node, step.s);
                    captured_edges += 1;
                }
            }
        }
    }
    if captured_edges != num_nodes * (num_nodes - 1) * sche_trees.len() {
        println!("WARNING: Schedule are Not Trees!");
    }
    Ok(sche_trees)
}

/// Outputs schedule to an MSCCL XML file
pub fn write_schedule(
    fname: &str,
    sche_trees: &[ScheduleTrees],
    network: &Network,
    collective: Collective,
) -> Result<(), Box<dyn Error>> {
    let ccl_sch = create_schedule(sche_trees, network, collective);
    fs::write(fname, &se::to_string(&ccl_sch)?)?;
    Ok(())
}

fn create_schedule(
    sche_trees: &[ScheduleTrees],
    network: &Network,
    collective: Collective,
) -> Algo {
    let num_nodes = network.node_count();
    let num_chunks = collective.phase_chunk_count(sche_trees.len());
    assert!(num_chunks > 0, "Cannot export an empty collective schedule");
    let schedule_chunks = num_nodes * num_chunks;
    let input_chunks = match collective {
        Collective::AllGather => num_chunks,
        Collective::AllReduce | Collective::ReduceScatter => schedule_chunks,
    };
    let mut ccl_sch = Algo {
        name: "l3sstree".to_string(),
        proto: "Simple".to_string(),
        nchannels: 1,
        nchunksperloop: schedule_chunks,
        ngpus: network.node_count(),
        coll: collective.msccl_name().to_string(),
        input_chunks: Some(input_chunks),
        chunk_layout: Some("subchunk-major".to_string()),
        inplace: 1,
        outofplace: 0,
        min_bytes: 0,
        max_bytes: 0,
        gpus: Vec::new(),
    };
    for node in 0..num_nodes {
        let mut ports: Vec<Port> = Vec::new();
        for (_, to_node, _) in network.edges_directed(node, Direction::Outgoing) {
            let port = Port {
                id: ports.len(),
                send: to_node.try_into().unwrap(),
                recv: -1,
                chan: 0,
                steps: Vec::new(),
            };
            ports.push(port);
        }
        for (from_node, _, _) in network.edges_directed(node, Direction::Incoming) {
            let port = Port {
                id: ports.len(),
                send: -1,
                recv: from_node.try_into().unwrap(),
                chan: 0,
                steps: Vec::new(),
            };
            ports.push(port);
        }
        let gpu = Gpu {
            id: node,
            i_chunks: match collective {
                Collective::AllGather => 0,
                Collective::AllReduce | Collective::ReduceScatter => schedule_chunks,
            },
            o_chunks: match collective {
                Collective::ReduceScatter => num_chunks,
                Collective::AllGather | Collective::AllReduce => schedule_chunks,
            },
            s_chunks: 0,
            ports: ports,
        };
        ccl_sch.gpus.push(gpu);
    }

    let time_range = sche_trees.iter().map(|sche| get_step_range(sche)).collect::<Vec<_>>();
    let max_time = time_range.iter().map(|srange| (*srange).1).max().unwrap();

    for time in 0..(max_time + 1) {
        for (schedule_index, sche) in sche_trees.iter().enumerate() {
            let logical_chunk = schedule_index % num_chunks;
            let is_reduction = collective.is_reduction_schedule(schedule_index, num_chunks);
            if time < time_range[schedule_index].0 || time > time_range[schedule_index].1 { continue; }
            for (schunk, sch_o) in sche.iter().enumerate() {
                let sch = sch_o.as_ref().unwrap();
                for (from_node, to_node, edge_t) in sch.all_edges() {
                    if *edge_t != time { continue; }
                    // from_node send
                    let workload_chunk = match collective {
                        Collective::AllGather => logical_chunk,
                        Collective::AllReduce | Collective::ReduceScatter =>
                            logical_chunk * num_nodes + schunk,
                    };
                    let mut send_step = Step {
                        s: time,
                        transact_type: "s".to_string(),
                        srcbuf: "o".to_string(),
                        srcoff: logical_chunk * num_nodes + schunk,
                        dstbuf: "o".to_string(),
                        dstoff: logical_chunk * num_nodes + schunk,
                        cnt: 1,
                        depid: -1,
                        deps: -1,
                        hasdep: 0,
                        logical_chunk: Some(workload_chunk),
                    };
                    // find parent node for depid and deps
                    if let Some((parent_node, _, _)) = 
                        sch.edges_directed(from_node, Direction::Incoming).next()
                    {
                        for port in ccl_sch.gpus[from_node].ports.iter() {
                            if port.recv != parent_node as isize { continue; }
                            for parent_step in port.steps.iter() {
                                if parent_step.dstoff == logical_chunk * num_nodes + schunk {
                                    send_step.depid = port.id as isize;
                                    send_step.deps = parent_step.s as isize;
                                    break;
                                }
                            }
                        }
                    }
                    for portid in 0..ccl_sch.gpus[from_node].ports.len() {
                        if ccl_sch.gpus[from_node].ports[portid].send == to_node as isize {
                            ccl_sch.gpus[from_node].ports[portid].steps.push(send_step);
                            break;
                        }
                    }

                    // to_node receive
                    let hasdep: u32 = match sch.edges_directed(to_node, Direction::Outgoing).next() {
                        Some(_) => 1,
                        None => 0,
                    };
                    let recv_step = Step {
                        s: time,
                        transact_type: (if is_reduction { "rrc" } else { "r" }).to_string(),
                        srcbuf: "o".to_string(),
                        srcoff: logical_chunk * num_nodes + schunk,
                        dstbuf: "o".to_string(),
                        dstoff: logical_chunk * num_nodes + schunk,
                        cnt: 1,
                        depid: -1,
                        deps: -1,
                        hasdep: hasdep,
                        logical_chunk: Some(workload_chunk),
                    };
                    for portid in 0..ccl_sch.gpus[to_node].ports.len() {
                        if ccl_sch.gpus[to_node].ports[portid].recv == from_node as isize {
                            ccl_sch.gpus[to_node].ports[portid].steps.push(recv_step);
                            break;
                        }
                    }
                }
            }
        }
    }
    ccl_sch
}

fn get_step_range(sche: &ScheduleTrees) -> (usize, usize) {
    let mut min_step: usize = usize::MAX;
    let mut max_step: usize = 0;
    for sch_o in sche.iter() {
        let sch = sch_o.as_ref().unwrap();
        for (_, _, step) in sch.all_edges() {
            if *step > max_step { max_step = *step; }
            if *step < min_step { min_step = *step; }
        }
    }
    (min_step, max_step)
}
