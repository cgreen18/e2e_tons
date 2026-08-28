# Third-Party Patch Escrow

The root repository deliberately keeps the upstream ASTRA-sim and CollectiveAPI
repositories as public submodules. Their TONS-specific modifications are stored
here as full binary-safe Git patches so this root commit remains accessible and
reconstructible even before personal forks are created.

These patches apply to the exact submodule revisions recorded in the root
commit:

| Patch | Upstream repository and base revision |
| --- | --- |
| `astra-network-analytical-tons.patch` | `astra-sim/astra-network-analytical` at `e8c5119f8d5a690b955e25c37a74359f23ac64cc` |
| `astra-sim-tons.patch` | `astra-sim/astra-sim` at `518bd513ae110428cd62eb60efc0f3993fd53c70` |
| `collectiveapi-tons.patch` | `astra-sim/collectiveapi` at `e1c2ef6b435e01cbf3675af5e38225c5113ba56a` |

Initialize the pinned submodules, then apply the patches from deepest repository
to parent:

```bash
git submodule update --init --recursive
git -C simul/astra-sim/extern/network_backend/analytical apply \
  ../../../../../third_party_patches/astra-network-analytical-tons.patch
git -C simul/astra-sim apply ../../third_party_patches/astra-sim-tons.patch
git -C simul/collectiveapi apply ../../third_party_patches/collectiveapi-tons.patch
```

The ASTRA parent patch intentionally excludes its nested analytical-backend
gitlink; the backend patch leaves that repository as a working-tree overlay.
This makes the sequence above work without depending on unpublished nested
commit IDs.

The preferred long-term publication step is to create forks under the project
owner, commit each patch to a `tons-e2e` branch, and update `.gitmodules` and
the parent gitlink to those public commits. GitHub fork creation could not be
authenticated from this environment (SSH push access is available, but no
GitHub API/CLI credential is installed), so patch escrow is the safe,
reproducible interim implementation.
