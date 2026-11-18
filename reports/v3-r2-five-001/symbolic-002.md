# Five representative weights on restored routes — symbolic-002

- Code commit A: `402e81fc3371db13db42fc7fec1d372d159d6d13`
- Base: `codex/v3-joint` / `accdedcaedae575405296604e3a398781282fc65`
- Run: `v3-r2-five-001/symbolic-002`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `08ba859255e04b5aa0afbe73b422694bdcfe2e8c988257fc1e33d0109f0d9faf`

Both frozen phases passed for the five representative weights: full norm, Q and FFN-down, actual embedding rows, and final 32 output-projection rows. All checked output bytes match native. Independent 44-token model check with four weights selected passed full first logits, native KV, positions and release. No performance claim.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
