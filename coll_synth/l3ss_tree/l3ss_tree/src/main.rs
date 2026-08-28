use std::error::Error;
use std::time;

use graph;
use build;
use schedule_view;
use msccl_view;
use pipe_view;
use pipe;
use dsl_io;

use clap::Parser;
use env_logger;
use petgraph::Direction;

#[derive(clap::ValueEnum, Copy, Clone, Debug)]
enum SchedulerType {
	/// Round Robin leaf scheduler
	Rr,
	/// Random leaf scheduler
	Rand,
}

impl SchedulerType {
	fn ident(&self) -> &str {
		match self {
			Self::Rr => "Round Robin",
			Self::Rand => "Random",
		}
	}
}

#[derive(clap::ValueEnum, Copy, Clone, Debug)]
enum CommType {
	/// All Reduce. Note that chunks * N communications will be scheduled.
	Ar,
	/// All Gather
	Ag,
	/// Reduce Scatter (the reduction phase of All Reduce)
	Rs,
}

impl CommType {
	fn ident(&self) -> &str {
		match self {
			Self::Ar => "All Reduce",
			Self::Ag => "All Gather",
			Self::Rs => "Reduce Scatter",
		}
	}
	fn convert_chunks(&self, chunks: usize, num_nodes: usize) -> usize {
		match self {
			Self::Ar | Self::Rs => chunks * num_nodes,
			Self::Ag => chunks,
		}
	}
	fn msccl_collective(&self) -> msccl_view::Collective {
		match self {
			Self::Ar => msccl_view::Collective::AllReduce,
			Self::Ag => msccl_view::Collective::AllGather,
			Self::Rs => msccl_view::Collective::ReduceScatter,
		}
	}
}

#[derive(clap::ValueEnum, Copy, Clone, Debug)]
enum ScheduleFType {
	Msccl,
	Pipe,
}

#[derive(clap::Subcommand, Clone, Debug)]
enum BuildType {
	/// Analyze stats on existing xml schedule
	View {
		/// Type of schedule file
		sch_type: ScheduleFType,
		/// Adj matrix of network topology
		topology: String,
		/// XML schedule file
		sch: String,
		/// If switched, pass in the switch start id
		#[arg(long)]
		switch: Option<usize>,
		/// Subchunking, used for dsl output. Default 100
		#[arg(default_value_t=100)]
		subchunk: usize,
	},
	/// Build schedule
	Build {
		/// Adj matrix of network topology
		topology: String,
		/// Type of communication
		#[arg(value_enum)]
		comm_type: CommType,
		/// Scheduler
		#[arg(value_enum, default_value_t=SchedulerType::Rr)]
		scheduler: SchedulerType,
		/// Number of chunks. If set to 0, will schedule r_in chunks.
		#[arg(default_value_t=0)]
		chunks: usize,
	},
	/// Build pipelined schedule
	Pipe {
		/// Adj matrix of network topology
		topology: String,
		/// If switched, pass in the switch start id
		#[arg(long)]
		switch: Option<usize>,
		/// Use legacy (convert port to node) switch removal
		#[arg(long, default_value_t=false)]
		legacy: bool,
		/// Specify number of chunks
		#[arg(long)]
		chunks: Option<usize>,
		/// Subchunking, used for dsl output. Default 100
		#[arg(long, default_value_t=100)]
		subchunk: usize,
	}
}

#[derive(Parser)]
#[command(name = "l3ss_tree")]
#[command(
	about = "Builds tree collective schedules for arbitrary network topologies.",
	long_about = None,
)]
struct CliArgs {
	/// Tree building method
	#[command(subcommand)]
	method: BuildType,

	/// Outputs visualization of networks in .svg file
	#[arg(long, value_name = "FILE")]
	see_network: Option<String>,

	/// Outputs visualization of built schedules in .html
	#[arg(long, value_name = "PATH")]
	see_schedule: Option<String>,

	/// Outputs schedule in MSCCL XML format
	#[arg(long, value_name = "XML_PATH")]
	xml: Option<String>,

	/// Outputs schedule in MSCCL++ DSL
	#[arg(long, value_name = "DSL_PATH")]
	dsl: Option<String>,

	/// Disables schedule checking. Note schedule checking disabled for pipelined by default.
	#[arg(long, action=clap::ArgAction::SetTrue)]
	no_check: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
	let args = CliArgs::parse();
	env_logger::init();

	let network_f = match args.method {
		BuildType::View { ref topology, .. } => topology,
		BuildType::Build { ref topology, .. } => topology,
		BuildType::Pipe { ref topology, .. } => topology,
	};

	let mut network = graph::digest_map(network_f)?;
	println!(
		"Network {} has {} nodes, {} edges, diameter {}, min in radix {}, out radix {}",
		network_f,
		network.node_count(),
		graph::count_tot_deg(&network),
		graph::algo::find_diameter(&network).len() - 1,
		graph::algo::find_radix(&network, Direction::Incoming),
		graph::algo::find_radix(&network, Direction::Outgoing),
	);

	let start_t = time::Instant::now();
	let mut built_collective: Option<msccl_view::Collective> = None;
	let sche = match args.method {
		BuildType::View { topology: _, ref sch_type, ref sch, .. } => {
			match sch_type {
				ScheduleFType::Msccl => msccl_view::build(sch)?,
				ScheduleFType::Pipe => pipe_view::build(sch)?,
			}
		},
		BuildType::Build { topology: _, comm_type, scheduler, mut chunks } => {
			built_collective = Some(comm_type.msccl_collective());
			if chunks == 0 { chunks = graph::algo::find_radix(&network, Direction::Incoming); }
			println!(
				"Scheduling {}, {} chunks, policy {}",
				comm_type.ident(),
				comm_type.convert_chunks(chunks, network.node_count()),
				scheduler.ident(),
			);
			let has_reduce = matches!(comm_type, CommType::Ar | CommType::Rs);
			let has_broadcast = matches!(comm_type, CommType::Ar | CommType::Ag);
			let (mut schedule, next_start_t) = if has_reduce {
				match scheduler {
					SchedulerType::Rr => build::build::<build::RRScheduler>(
						&mut network,
						chunks,
						Direction::Incoming,
						0
					),
					SchedulerType::Rand => build::build::<build::RandScheduler>(
						&mut network,
						chunks,
						Direction::Incoming,
						0
					),
				}
			} else {
				(Vec::new(), 0)
			};
			if has_reduce {
				build::reverse_schedule(&mut schedule, 0, next_start_t - 1);
			}
			if has_broadcast {
				let broadcast_sch = match scheduler {
					SchedulerType::Rr => build::build::<build::RRScheduler>(
						&mut network,
						chunks,
						Direction::Outgoing,
						next_start_t,
					).0,
					SchedulerType::Rand => build::build::<build::RandScheduler>(
						&mut network,
						chunks,
						Direction::Outgoing,
						next_start_t,
					).0,
				};
				schedule.extend(broadcast_sch);
			}
			schedule
		},
		BuildType::Pipe { topology: _, ref switch, legacy, chunks, .. } => {
			pipe::build(&network, *switch, legacy, chunks)
		}
	};
	println!("Scheduling took {}s", start_t.elapsed().as_nanos() as f64 / 1_000_000_000.0);

	match args.method {
		BuildType::View { ref sch_type, .. } => {
			match sch_type {
				ScheduleFType::Msccl => schedule_view::stat(&sche, &network, args.no_check, false),
				ScheduleFType::Pipe => schedule_view::stat(&sche, &network, args.no_check, true),
			}
		},
		BuildType::Pipe { .. } => schedule_view::stat(&sche, &network, args.no_check, true),
		BuildType::Build { .. } => 
			schedule_view::stat(&sche, &network, args.no_check, false),
	}

	if let Some(net_vis) = args.see_network {
		println!("Writing network...");
		graph::visualize_graph(&network, &net_vis)?;
	}
	if let Some(sche_vis) = args.see_schedule {
		println!("Writing schedule...");
		for (i, sch) in sche.iter().enumerate() {
			graph::visualize_schedule(&sch, &format!("{}/{}.html", sche_vis, i))?; 
		}
	}
	if let Some(xml_f) = args.xml {
		println!("Writing MSCCL...");
		let collective = built_collective.ok_or(
			"MSCCL export of an imported/viewed schedule requires collective metadata"
		)?;
		msccl_view::write_schedule(&xml_f, &sche, &network, collective)?;
	}
	if let Some(dsl_f) = args.dsl {
		println!("Writing MSCCL++...");
		match args.method {
			BuildType::Pipe { ref subchunk, ref switch, .. } => {
				let slim = match switch {
					Some(sid) => *sid,
					None => network.node_count(),
				};
				dsl_io::write_pipe(&sche, slim, *subchunk, &dsl_f)?;
			},
			BuildType::View { ref sch_type, ref switch, ref subchunk, .. } => {
				match sch_type {
					ScheduleFType::Pipe => {
						let slim = match switch {
							Some(sid) => *sid,
							None => network.node_count(),
						};
						dsl_io::write_pipe(&sche, slim, *subchunk, &dsl_f)?;
					},
					_ => dsl_io::write_step(&sche, &dsl_f)?,
				}
			},
			BuildType::Build { .. } => dsl_io::write_step(&sche, &dsl_f)?,
		}
	}
	Ok(())
}
