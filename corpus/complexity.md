# Complexity analysis

Complexity describes how the cost of an algorithm grows with the size of its
input, ignoring constant factors and hardware.

## Big-O, Omega and Theta

Big-O is an upper bound: f(n) = O(g(n)) means f grows no faster than g beyond
some input size. Omega is the matching lower bound, and Theta means both hold,
so the growth rate is pinned exactly.

Saying quicksort is O(n^2) is true but weak; saying its worst case is
Theta(n^2) says the bound is tight.

## Amortised analysis

Amortised cost averages an expensive operation over the cheap ones that make it
rare. Appending to a dynamic array is O(1) amortised: most appends write one
slot, and the occasional resize copies n elements but doubles the capacity, so
resizes happen exponentially less often. Any single append can still be O(n) --
amortised is not the same as average, and it is not a guarantee about latency.

## Why a slow loop is often not the loop

A loop that looks linear can hide a linear operation inside it. Searching a
list inside a loop over that list is O(n^2); the same code against a set or
dict is O(n), because membership testing drops from linear to constant. When
code is slower than its structure suggests, the usual cause is a data structure
whose operation costs more than it appears to, not the loop itself.

## Space-time trade

Memoisation trades space for time by storing results instead of recomputing
them. It only pays when the same inputs recur and the stored set stays bounded;
an unbounded cache turns a time problem into a memory problem.
