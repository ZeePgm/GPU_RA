# Five representative weights on restored routes — cuda-001

- Code commit A: `a5201f63b8c622a13cbd003602f827c7820955a2`
- Base: `codex/v3-joint` / `accdedcaedae575405296604e3a398781282fc65`
- Run: `v3-r2-five-001/cuda-001`
- Process status / exit: `completed` / `0`
- Author verdict: `passed`
- Data manifest SHA-256: `4e02490f81433a651573f99ce5bb93b6617300a2aa5a4cf0babb286acd179a0d`

Both frozen phases passed for the five representative weights: full norm, Q and FFN-down, actual embedding rows, and final 32 output-projection rows. All checked output bytes match native. Independent 44-token model check with four weights selected passed full first logits, native KV, positions and release. No performance claim.

Traceability checks do not replace scientific acceptance. Commit this report separately; its code reference remains A.
