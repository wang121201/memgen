# RTX 4000 Ada HBFSim configuration snapshot

`rtx4000-ada-uncalibrated.cfg` is vendored from the v2-compatible HBFSim
overlay at:

`/home/xmu/nvidiagds/simulators/HBFSim/configs/overlays/gddr6/rtx4000-ada-uncalibrated.cfg`

Its SHA-256 is
`b0b1098dd31828b92c11a8ec9d8eca8b61ccae5a2389e1f8612aba967a41a1ca`.
The snapshot targets HBFSim source commit
`d7a2ca64614a6d9ce8d7a69beb77ce78b66df1a8`.

`eight-stack-baseline.cfg` is the parser-complete system base from that clean
source commit; its SHA-256 is
`a1e9e46f7054260351d04101fd54c1bf4df144cd606fc88fe5519d26390c2823`.
The runner applies the base first and the GDDR6 overlay second. HBF remains
disabled at execution, so the base's HBF geometry supplies schema identity but
does not generate HBF work or traffic.

The configuration uses the generic banked HBM engine as an explicit,
uncalibrated GDDR6 service model. Board capacity, aggregate interface width
and peak-rate arithmetic do not establish the real channel/bank/row mapping,
controller policy, refresh behavior or service timing. Results using this file
are sensitivity diagnostics until matched-scope hardware calibration exists.

An older profile tied to transaction protocol v1 used
`hbm-compiled-span-acceleration`; source commit `d7a2ca6` no longer owns that
key. The incompatible v1 configuration is intentionally not part of this
branch and must not be mixed with the pinned v2 executable.
