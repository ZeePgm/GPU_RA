# Five representative weights on restored routes — cuda-002

- Code commit A: `402e81fc3371db13db42fc7fec1d372d159d6d13`
- Base: `codex/v3-joint` / `accdedcaedae575405296604e3a398781282fc65`
- Run: `v3-r2-five-001/cuda-002`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `8ed169bc82c998e8f0b5443faf1cb6b49a80cd1a3bd4ce32c12318ebf4bd2ffd`

Both frozen phases passed for the five representative weights: full norm, Q and FFN-down, actual embedding rows, and final 32 output-projection rows. All checked output bytes match native. Independent 44-token model check with four weights selected passed full first logits, native KV, positions and release. No performance claim.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
