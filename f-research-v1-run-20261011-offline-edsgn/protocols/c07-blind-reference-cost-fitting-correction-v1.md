# C07: periodic/random reference cost allocation

Prospective root finding during P3. No V5/V6 formal prediction, calibration or
profile has run, and no final outcome motivated this correction. The active
package, source-bound training campaign and queues remain unchanged.

In scheduling.Policy._select, periodic and random optional proposals use their
own blind orders. But the common _fit_cost_budget subsequently sorts offered
nodes using sg/ss, which contain residual/quality scores. This makes a budget-
constrained reference partly adaptive and cannot support a pure periodic or
random comparison as written. Mandatory actions and quota logic are otherwise
retained. Full-policy baseline training does not use this selective branch.

Prepare an isolated instance-level adapter for periodic/random only. Preserve
the dataclass fields, ceil(N*fraction) independent quotas, offered nodes,
initialization/age/audit mandatory sets, measured B1 table and shared budget.
Replace the optional cost-allocation ranking with each method's blind node
order, interleaving graph/state ranks normalized by their independent quotas.
Use deterministic graph/state/node tie breaking and grouped incremental cost
checks. Mandatory excess is executed and reported. V7 adaptive ranking and full
policy behavior are unchanged. Changing scores alone must not change adapted
V5/V6 output masks for identical runtime and proposals.

Before use, accept source/toy evidence independently and root-review the
adapter. Every frozen prediction/robustness/profile entry must load the same
adapter, bind its exact source/rule version in freeze extras, verify the binding
before and after execution, and record the rule. A stored policy dataclass hash
alone is insufficient to identify the changed procedure. Original predictions
and metrics must not be mislabeled or silently reused; no affected result
currently exists. The managed launcher still binds all predecessor and source
bytes. This is a control-isolation correction within the RP method family,
not a new architecture or another pilot search candidate.

Status: open; helper, producer integration and actual matched execution pending.
