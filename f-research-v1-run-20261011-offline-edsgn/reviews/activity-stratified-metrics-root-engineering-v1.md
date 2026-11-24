# Root review: training activity strata reporting helper

Three focused CPU toy tests pass, including masked raw-cell pooling and equal
horizon/equal seed aggregation, training ties, empty groups, zero versus
unobserved training nodes and signed synthetic magnitudes. Root inspected the
pure implementation. The reporting rule is prospective and cannot fit thresholds
on test data. A real guarded caller must still bind the actual training slice,
node order, frozen strata file and reporting inputs. No actual strata fit or test
report ran. This engineering record grants no phase, inference or test authority.
