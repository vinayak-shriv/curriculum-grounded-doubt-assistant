# Database indexing

An index is a separate structure that lets the engine find rows without
scanning the table. It speeds reads and slows writes, since every insert,
update and delete must maintain it.

## Clustered and non-clustered

A clustered index determines the physical order of the rows in the table, so a
table can have at most one. Range queries over the clustered key are fast
because the matching rows are physically adjacent.

A non-clustered index is a separate structure holding the indexed columns plus
a pointer back to the row. Reading columns not in the index requires following
that pointer, a step usually called a bookmark lookup, which is why a
non-clustered index can be ignored by the planner when the query returns many
rows -- at some point scanning the table beats doing lookups one at a time.

## Composite indexes and column order

In an index on (a, b), the order is not cosmetic. The index can serve a
predicate on a, or on a and b together, but not one on b alone, because rows
are sorted by a first. This is the leftmost-prefix rule and it is the single
most common reason an index that "should" be used is not.

## Covering indexes

An index that contains every column a query needs lets the engine answer from
the index alone and skip the table entirely. This turns a bookmark lookup per
row into nothing, at the cost of a wider index to maintain.

## When an index is not used

The planner will skip an index when the query is not selective enough, when a
function is applied to the indexed column (WHERE YEAR(created) = 2024 cannot
use an index on created), or when statistics are stale enough that the
estimated row count is wrong.
