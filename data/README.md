# data/

Evaluation examples are not stored here: they are regenerated deterministically from
`(seed, task, split, difficulty)` by `src/jevq/tasks`, and each run record carries a SHA-256
fingerprint of the exact examples used.

Reserved for artifacts that cannot be regenerated cheaply:

- `train/` materialised training splits for V1 and later (so a run can cite a fixed file);
- `traces/` teacher reasoning traces for V5.
