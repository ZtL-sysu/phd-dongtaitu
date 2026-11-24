# Candidate full-path probe prelaunch review v1

**Decision: do not launch this GPU probe yet.** Candidate representation parity is promising, and the CPU full-model toy run is exact, but the current probe can lose failure evidence and its reported interval omits input preparation/transfer. It also lacks a pre/post source-and-data fence and a non-raceable GPU serialization guard.

## Evidence supporting the candidate path

- `numpy_contacts` in `candidate_gpu_fullpath_probe_v1.py:23-34` implements the same `(-count,node_id)` top-K rule as `build_contact_matrix` in the CPU v2 diagnostic. A direct toy fixture with empty rows, tied priorities and sparse contacts produced identical ID/count/valid matrices.
- The stored CPU v2 diagnostic at `logs/candidate-bounded-tensor-diagnostic-v2.json` compares candidate IDs, frequencies and candidate counters for full/partial rows, changing contacts across steps, phase transitions, V0, and N1/K0. All mismatches are zero; its bounded merge uses `[rows,K,K]` and does not form N-by-N matrices.
- A separate CPU-only full model check (B=2,L=6,N=7,F=2,H=3) patched only `_candidates`, used identical initial weights and event-prior windows, and compared predictions, graph loss, every non-null parameter gradient, and the full serialized counter dictionary. All were exactly equal (max differences 0); outputs and gradients were finite. No data bundle, GPU, optimizer or training run was used. Details: `logs/candidate-fullpath-cpu-toy-v1.json`.

## Blocking issues before GPU launch

1. **Failure evidence and budget accounting are absent.** The script writes its JSON only at `candidate_gpu_fullpath_probe_v1.py:129`, after both arms complete. Any OOM, assertion, or process interruption leaves no receipt with elapsed time or partial measurements. There is no cumulative-budget check or failure-safe partial record. Create an exclusive receipt before loading data, update it after each arm/pass, and finalize status in `finally`; charge elapsed wall time even on failure.
2. **Timing does not include full input preparation/transfer.** The data is selected and `batch_at(..., device='cuda')` runs at line 63 before either arm timer starts. The timer at lines 79-84 therefore measures the model forward/backward with already-resident CUDA inputs. Candidate contact construction and its internal `.to(cuda)` are inside the proposal's forward and are timed, but batch assembly/host-to-device transfer is excluded. Either explicitly name the metric model-only and report prep/transfer separately, or include equivalent per-arm prep and transfer in an end-to-end metric.
3. **No immutable-source/data fence.** The report records only a post-run `model.py` hash at line 125. It does not record a pre-run hash, hash the probe/diagnostic/scheduling/engine/common/config inputs, bind the prepared train bundle, or verify inputs remained unchanged. Capture before hashes and verify them before writing success; on mismatch preserve a failed receipt.
4. **The single process scan is not a lock.** The check at line 47 can race with another owned job starting after the check and only detects the enumerated process patterns. Add a shared exclusive resource lock (or make launch contingent on the existing campaign lock/owner) and recheck immediately before CUDA allocation. A root-level confirmation that the V1 process has ended is still required; no GPU job was launched by this review.
5. **GPU-path equivalence checks are narrower than the CPU evidence.** At lines 94-104, full-model parity checks only six counters and compares model outputs/loss/gradients on the third pass. Prefer exact full serialized counter-dictionary parity, plus direct candidate ID/frequency parity at the patched boundary, to make drift diagnosable. CPU v2 already supports the expected empty/V0/N1/partial/phase fixtures, so this need not become a full model suite.

The two arms do use the same eight selected training origins and cloned initial state (`:59-76`), with one marked warmup and the last pass used for parity. The diagnostic is correctly labelled as not an inference speedup/energy claim. These positives do not resolve the launch blockers above.

## Run record and source hashes

CPU toy scope: resource/functionality diagnostic, wall 0.101 s inside the check body (about 4.4 s process wall including Python startup); CPU only, no real dataset, optimizer, or GPU. The GPU script itself was inspected but not executed.

- `edsgn/scripts/candidate_gpu_fullpath_probe_v1.py`: `0ac8e659dbd6d54811b1c227a4b35ac91618fd1177210b328e9237595263e3e5`
- `edsgn/scripts/candidate_bounded_tensor_diagnostic_v2.py`: `e261a34a75f93357be72d3260bf084dd6a41135b2f4dcf68d5396782e4631333`
- `edsgn/src/edsgn/model.py`: `b60c7e3c4c961ed87ba9b4d9fac604f63f7c9b2aef351bba18741921f6b97d9c`
- `logs/candidate-bounded-tensor-diagnostic-v2.json`: `80a2513d01a86378988718294b304836e7402231db2e4722e83f5ef1a82485d9`
