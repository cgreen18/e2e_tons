#!/usr/bin/env python3
"""
Generate an MSCCL XML schedule for MSCCLang's "allpairs/direct" AllToAll.

This reproduces the structure observed in MSCCLang's emitted XML:
- One <algo> root
- One <gpu> per rank
- For each GPU g:
    * (n-1) recv threadblocks (tb ids 0..n-2), one per peer p != g
    * (n-1) send threadblocks (tb ids n-1..2n-3), one per peer p != g
    * The *first* send TB (tb id n-1) also includes a local self-copy:
        input[g] -> output[g]

Semantics (matches example XMLs):
- AllToAll transposes: chunk i on GPU j ends up on GPU i at index j.
- Send TB on GPU g to peer p sends chunk p from g's input; receiver writes it at dstoff=g.
- Recv TB on GPU g from peer p receives chunk g from p's input; writes it at dstoff=p.

This generator is O(n^2) in time and output size (unavoidable, since the schedule itself is O(n^2)),
but it avoids MSCCLang's compilation overhead (DAG construction + lowering) and streams output.

Example:
  ./gen_alltoall_allpairs_mscccl.py -n 10 -o ata_10n_1c.xml
  ./gen_alltoall_allpairs_mscccl.py -n 1024 -c 8 --gzip -o ata_1024n_8c.xml.gz
"""

from __future__ import annotations

import argparse
import gzip
from typing import TextIO, Optional


def _chan_for_peer(peer: int, nchannels: int) -> int:
    # Deterministic mapping. For nchannels==1 this matches MSCCLang examples.
    # For nchannels>1, this distributes point-to-point TBs across channels by peer id.
    return peer % nchannels


def write_alltoall_allpairs(
    n: int,
    nchannels: int,
    proto: str,
    inplace: int,
    algo_name: str,
    out_f: TextIO,
    attach_self_copy: bool = True,
) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if nchannels <= 0:
        raise ValueError("nchannels must be positive")
    if inplace not in (0, 1):
        raise ValueError("inplace must be 0 or 1")

    out_f.write(
        f'<algo name="{algo_name}" proto="{proto}" nchannels="{nchannels}" '
        f'nchunksperloop="{n}" ngpus="{n}" coll="alltoall" inplace="{inplace}">\n'
    )

    # Emit one GPU at a time to keep memory bounded.
    for g in range(n):
        lines = []
        lines.append(f'  <gpu id="{g}" i_chunks="{n}" o_chunks="{n}" s_chunks="0">\n')

        tb_id = 0

        # ----
        # Recv TBs: from every peer p != g, receive peer's chunk g into output at index p.
        # ----
        for peer in range(n):
            if peer == g:
                continue
            chan = _chan_for_peer(peer, nchannels)
            lines.append(f'    <tb id="{tb_id}" send="-1" recv="{peer}" chan="{chan}">\n')
            lines.append(
                f'      <step s="0" type="r" srcbuf="i" srcoff="{g}" '
                f'dstbuf="o" dstoff="{peer}" cnt="1" depid="-1" deps="-1" hasdep="0"/>\n'
            )
            lines.append('    </tb>\n')
            tb_id += 1

        # ----
        # Send TBs: to every peer p != g, send chunk p; receiver writes it at output index g.
        # Also attach the local copy input[g] -> output[g] to the first send TB.
        # ----
        first_send = True
        for peer in range(n):
            if peer == g:
                continue
            chan = _chan_for_peer(peer, nchannels)
            lines.append(f'    <tb id="{tb_id}" send="{peer}" recv="-1" chan="{chan}">\n')
            lines.append(
                f'      <step s="0" type="s" srcbuf="i" srcoff="{peer}" '
                f'dstbuf="o" dstoff="{g}" cnt="1" depid="-1" deps="-1" hasdep="0"/>\n'
            )
            if attach_self_copy and first_send:
                lines.append(
                    f'      <step s="1" type="cpy" srcbuf="i" srcoff="{g}" '
                    f'dstbuf="o" dstoff="{g}" cnt="1" depid="-1" deps="-1" hasdep="0"/>\n'
                )
                first_send = False
            lines.append('    </tb>\n')
            tb_id += 1

        lines.append('  </gpu>\n')
        out_f.write("".join(lines))

    out_f.write("</algo>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MSCCL XML for AllToAll allpairs/direct.")
    ap.add_argument("-n", "--ngpus", type=int, required=True, help="Number of GPUs/ranks.")
    ap.add_argument("-c", "--nchannels", type=int, default=1, help="Number of channels (default: 1).")
    ap.add_argument("--proto", type=str, default="Simple", help='Protocol string (e.g., "Simple", "LL", "LL128").')
    ap.add_argument("--inplace", type=int, default=0, choices=[0, 1], help="Set inplace attribute (0 or 1).")
    ap.add_argument("--name", type=str, default="alltoall_allpairs", help="Algo name attribute.")
    ap.add_argument("-o", "--out", type=str, required=True, help="Output path (.xml or .xml.gz if --gzip).")
    ap.add_argument("--gzip", action="store_true", help="Gzip-compress the output.")
    ap.add_argument(
        "--no-self-copy",
        action="store_true",
        help="Do not emit the local self-copy step (MSCCLang examples DO include it).",
    )

    args = ap.parse_args()

    attach_self_copy = not args.no_self_copy

    if args.gzip or args.out.endswith(".gz"):
        with gzip.open(args.out, "wt", encoding="utf-8", newline="\n") as f:
            write_alltoall_allpairs(
                n=args.ngpus,
                nchannels=args.nchannels,
                proto=args.proto,
                inplace=args.inplace,
                algo_name=args.name,
                out_f=f,
                attach_self_copy=attach_self_copy,
            )
    else:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            write_alltoall_allpairs(
                n=args.ngpus,
                nchannels=args.nchannels,
                proto=args.proto,
                inplace=args.inplace,
                algo_name=args.name,
                out_f=f,
                attach_self_copy=attach_self_copy,
            )


if __name__ == "__main__":
    main()
