# Global completion v4 delta review

Verdict: pass for the narrow v3→v4 source delta; engineering metadata only.

The diff is limited to two changes: simple_v3 and case_v3 now fail when their phase receipt omits its full output hash map; prediction_v5 now rejects duplicate paths after RUN-path resolution, including a repeated identical ref. The mixed seven-driver fixture covers these attacks and retains graph/H2 branches unchanged.

Focused Conda tests: 4 passed, 30 deselected; whole command 2.28s. Log SHA256 `aa9de60cc98f31ef4530d078b9009a84b20155a7c0ca4d0771af9aee1e27d0f6`. Source SHA256s are pinned in the companion JSON.

No real arrays, model/checkpoint, oracle, GPU, queue, or completed campaign artifacts were inspected. This is a delta review only; it does not establish scientific acceptance or completeness beyond the registered seven-driver contracts.
