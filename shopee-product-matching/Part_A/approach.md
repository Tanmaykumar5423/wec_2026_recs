# Approach — Part A

## Questions
- How many listings and product groups exist?
- How large are groups?
- How many image references are unique or repeated?
- How noisy are titles?
- Which examples have conflicting text and visual evidence?

## Analysis
1. Inspect files and columns.
2. Compute listing/group/image statistics.
3. Analyze title length, exact duplicates and near duplicates.
4. Visualize group-size distribution.
5. Display curated hard cases:
   - same product, different titles
   - same product, different images
   - similar titles, different products
   - visually similar, different products
   - missing/noisy metadata

Every chart gets an observation and a modeling implication.
