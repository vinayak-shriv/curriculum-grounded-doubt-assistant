# Sorting

Sorting arranges records into a defined order. The choice of algorithm is
usually a trade between worst-case guarantees, memory, and whether the order of
equal elements has to be preserved.

## Stability

A sort is stable if records comparing equal keep their original relative order.
This matters when sorting by one key after another: sorting by name and then
stably by department leaves each department's names still in order. Merge sort
is stable; quicksort and heapsort are not.

## Merge sort

Merge sort divides the array in half, sorts each half recursively, and merges
the two sorted halves in linear time.

It runs in O(n log n) in the best, average and worst case, because the division
is independent of the data. The cost is O(n) auxiliary space for the merge
buffer, which is what rules it out in memory-constrained settings.

## Quicksort

Quicksort picks a pivot, partitions the array so that smaller elements sit left
of it and larger ones right, and recurses on each side.

The average case is O(n log n) and the constant factor is small, which is why
it is the usual default. The worst case is O(n^2), reached when the pivot
repeatedly lands at an extreme of the range -- classically, choosing the first
element as pivot on input that is already sorted. Randomised or median-of-three
pivot selection makes that case vanishingly unlikely without changing the
worst-case bound.

Quicksort sorts in place, using only O(log n) stack space for the recursion.

## Heapsort

Heapsort builds a binary heap and repeatedly extracts the maximum. It is
O(n log n) worst case and sorts in place, so it has better guarantees than
quicksort and better memory behaviour than merge sort. It loses on constant
factors and cache locality, which is why it is rarely the default despite
looking best on paper.
