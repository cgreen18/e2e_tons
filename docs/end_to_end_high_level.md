# End-to-End AI Interconnect Evaluation

*Brief design for public workloads, Chakra, ASTRA-sim, MSCCLang, and analytical network backends*

**Status:** Proposed research workflow  
**Updated:** August 26, 2026

> **Recommended flow:** Represent model execution as a per-rank Chakra DAG, lower each collective into an MSCCLang-derived point-to-point schedule, run both analytical variants for fast screening and contention-aware end-to-end results, and reserve a packet/flit backend for routing and deadlock validation.

## 1. End-to-end flow

1. **Select a public workload source.** Start with a real Chakra execution trace (ET) when available. Otherwise, use a public benchmark implementation and collect PyTorch/Chakra traces for one representative iteration after warm-up.

2. **Normalize to Chakra ET.** Produce one trace per rank containing compute, memory, collective, and point-to-point nodes plus data and control dependencies. Preserve measured compute durations initially, then add compute-time scaling experiments to reduce dependence on the source GPU.

3. **Select and lower collective plans.** Index plans by collective type, participant group, message-size range, and objective (low latency or high throughput). Compile MSCCLang to MSCCL-IR and convert the algorithm into per-rank Chakra communication DAGs containing sends, receives, and optional local-compute operations.

4. **Compose workload and collective DAGs.** When ASTRA-sim reaches a collective node, instantiate the selected per-rank schedule and release its operations according to dependencies. The collective completes when the schedule DAG completes, after which dependent model nodes become ready.

5. **Simulate the candidate interconnect.** Provide the same topology, link parameters, routing table, workload trace, and schedule library to each experiment. Run the congestion-unaware backend for rapid screening and the congestion-aware backend for shared-link contention.

6. **Collect paired results.** Compare the baseline torus, twisted torus, and proposed topology/routing variants using identical workload and collective inputs. Report iteration time, exposed communication time, collective critical-path time, throughput, and instrumented per-link utilization and queueing.

## 2. Public trace and workload sources

Public repositories more often provide schemas, generators, profilers, and runnable workload definitions than complete production-scale Chakra traces. The workflow should therefore support both direct trace ingestion and trace collection from public implementations.

| Source | Recommended use | Caveat |
|---|---|---|
| **MLCommons Chakra** | Use sample or generated ETs to validate converters, dependencies, rank mapping, and ASTRA execution. | Examples are primarily functional; do not treat them as a representative model suite. |
| **PARAM / PyTorch ET** | Capture operator-level ET and Kineto timing from public PyTorch workloads. DLRM is useful as an all-to-all-heavy stressor. | Trace capture and linking can require version-specific cleanup and validation. |
| **MLPerf Training** | Use current dense LLM and MoE benchmark implementations as reproducible trace-generation targets. | The repository supplies models and configurations, not generally ready-to-simulate Chakra ETs. |
| **Megatron-LM / DeepSpeed** | Generate dense and expert-parallel traces with controlled tensor, pipeline, data, context, and expert parallelism. | Record the exact software commit, parallel configuration, batch and sequence sizes, and timing hardware. |

## 3. MSCCLang and ASTRA-sim integration

The ASTRA-sim Collective API repository demonstrates the intended workflow: MSCCLang produces MSCCL-IR, a converter represents the collective algorithm as Chakra ET, and ASTRA-sim consumes both the model workload ET and collective-algorithm ET. Treat this repository as a reference branch and port its converter and API to the exact ASTRA-sim revision used for the experiments.

- Maintain separate workload ET and collective-schedule ET artifacts so the same model trace can be evaluated with multiple schedules and topologies.

- Choose the plan at each collective call using collective type, bytes per rank, communicator membership, and the latency/throughput policy.

- Preserve compute-communication overlap through Chakra dependencies and ASTRA resource availability; do not serialize all compute and communication globally.

- Validate every lowered schedule for matching send/receive byte counts, reachable dependencies, and completion on all ranks before performance simulation.

## 4. Analytical backend roles

| Backend | What it models | Role in the study |
|---|---|---|
| **Congestion-unaware** | Independent transfer delay, approximately $H \times \text{link latency} + \text{bytes}/\text{bandwidth}$. Concurrent transfers do not interact. | Fast functional validation, topology and schedule sweeps, compute-overlap sensitivity, and an optimistic no-contention baseline. |
| **Congestion-aware** | Explicit directed links and fixed routes. Whole chunks queue FIFO and serialize when transfers share a link. | Primary analytical end-to-end result for topology and static-routing contention. Instrument link busy time and queue delay. |

### Required congestion-aware extension

Add a `GraphTopology` that loads arbitrary directed links and source-destination routes. The current implementation is limited to one-dimensional Ring, Switch, and FullyConnected configurations.

A complete ASTRA send is modeled as a whole store-and-forward chunk; the analytical backend has no packetization, finite buffers, virtual channels, credit backpressure, or routing deadlock. Therefore, use it for application-level contention—not as the sole validation of the proposed deadlock-avoidance method. Validate routing and deadlock separately with a packet/flit-level model.

## 5. Minimum experiment matrix

- **Workloads:** At least one dense LLM and one MoE/expert-parallel workload, plus an isolated collective suite.

- **Scales:** 64, 128, and 256 endpoints when trace scaling and communicator construction remain valid.

- **Plans:** Latency-optimized and throughput-optimized MSCCLang schedules selected by message-size range.

- **Networks:** Baseline torus, twisted torus, and each proposed topology/routing combination with identical link budgets.

- **Backends:** Run all cases congestion-unaware; run representative and final cases congestion-aware; validate routing and deadlock separately in a packet/flit model.

## Repository URLs

- [ASTRA-sim](https://github.com/astra-sim/astra-sim)
- [ASTRA-sim Collective API](https://github.com/astra-sim/collectiveapi)
- [Chakra](https://github.com/mlcommons/chakra)
- [MSCCL-tools / MSCCLang](https://github.com/microsoft/msccl-tools)
- [ASTRA analytical network backend](https://github.com/astra-sim/astra-network-analytical)
- [PARAM](https://github.com/facebookresearch/param)
- [MLPerf Training](https://github.com/mlcommons/training)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples)
- [DLRM](https://github.com/facebookresearch/dlrm)
