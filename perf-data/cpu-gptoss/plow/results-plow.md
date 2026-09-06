# bench-api results

- server: `http://localhost:8096`  model: `6cee5e81ee83917806bbde320786a8fb61efebee`
- time: 2026-09-06T02:52:42+00:00  seed: 1234  max_tokens: 64  warmup: 1
- prompt tokens counted by: tokenizer
- server models: 6cee5e81ee83917806bbde320786a8fb61efebee

| workload | conc | n | err | in tok | out tok | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | latency p50/p90 s | out tok/s | req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| chat_short | 1 | 8 | 0 | 32 | 64.0 | 1128/1180/1294 | 55/54/55 | 4.40/4.70 | 14.0 | 0.22 |
| chat_short | 2 | 8 | 0 | 32 | 64.0 | 1739/1881/2291 | 82/86/93 | 7.01/7.39 | 18.4 | 0.29 |
| chat_short | 4 | 8 | 0 | 32 | 64.0 | 2384/2715/2892 | 156/160/170 | 12.60/12.89 | 20.8 | 0.32 |
| chat_short | 8 | 8 | 0 | 32 | 64.0 | 5118/5238/8394 | 281/294/320 | 22.95/23.38 | 21.7 | 0.34 |
| chat_long | 1 | 8 | 0 | 392 | 58.4 | 9067/11117/12136 | 48/48/49 | 13.12/15.18 | 4.9 | 0.08 |
| chat_long | 2 | 8 | 0 | 392 | 53.1 | 11282/12124/15360 | 203/164/310 | 25.30/31.11 | 4.5 | 0.08 |
| chat_long | 4 | 8 | 0 | 392 | 53.1 | 19181/17900/30114 | 510/741/837 | 41.17/68.26 | 4.6 | 0.09 |
| chat_long | 8 | 8 | 0 | 392 | 53.1 | 41081/39330/69122 | 854/995/1165 | 91.66/92.97 | 4.5 | 0.09 |
| code | 1 | 8 | 0 | 344 | 64.0 | 7094/6929/8519 | 49/49/49 | 10.06/11.61 | 6.3 | 0.10 |
| code | 2 | 8 | 0 | 344 | 64.0 | 9405/9048/13010 | 168/173/194 | 19.92/25.13 | 6.4 | 0.10 |
| code | 4 | 8 | 0 | 344 | 64.0 | 15989/15491/21500 | 358/408/418 | 40.92/47.79 | 6.6 | 0.10 |
| code | 8 | 8 | 0 | 344 | 64.0 | 32703/33986/54546 | 696/791/983 | 76.71/78.46 | 6.5 | 0.10 |
| summarize | 1 | 8 | 0 | 1111 | 64.0 | 25789/28088/33973 | 50/50/51 | 31.31/37.13 | 2.2 | 0.03 |
| summarize | 2 | 8 | 0 | 1111 | 64.0 | 33762/34446/47883 | 422/388/608 | 56.06/71.68 | 2.1 | 0.03 |
| summarize | 4 | 8 | 0 | 1111 | 64.0 | 46203/35346/69748 | 1170/1478/1497 | 125.77/162.99 | 2.1 | 0.03 |
| summarize | 8 | 8 | 0 | 1111 | 64.0 | 118568/138075/196529 | 1874/2190/3093 | 240.31/244.30 | 2.1 | 0.03 |

## Samples

**chat_short @ 1**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s needs s`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 2**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s needs s`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 4**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s needs s`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_short @ 8**

- #0 chat_short in=25 out=64 fin=length: `At the end of summer, the queen’s egg‑laying rate drops and the colony’s needs s`
- #1 chat_short in=22 out=64 fin=length: `Below is a concise comparison of the two most common coffee processing methods—*`

**chat_long @ 1**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**  An escapement is a mechanical device that controls the release of en`

**chat_long @ 2**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that controls the release of e`

**chat_long @ 4**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that controls the release of e`

**chat_long @ 8**

- #0 chat_long in=506 out=64 fin=length: `**Where to keep it:**   A separate savings account that takes a day to transfer `
- #1 chat_long in=383 out=64 fin=length: `**Answer**   An escapement is a mechanical device that controls the release of e`

**code @ 1**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Logs 0, 1, 2 after one, two, and three seconds.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /* 1️⃣  Use explicit JOIN syntax – it is clearer and less e`

**code @ 2**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Logs 0, 1, 2 after one, two, and three seconds.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /* 1️⃣  Use explicit JOIN syntax and LEFT JOIN to keep cust`

**code @ 4**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Logs 0, 1, 2 after one, two, and three seconds.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /* 1️⃣  Use explicit JOIN syntax and LEFT JOIN to keep cust`

**code @ 8**

- #0 code in=441 out=64 fin=length: `'''javascript /**  * Logs 0, 1, 2 after one, two, and three seconds.  * Uses 'le`
- #1 code in=284 out=64 fin=length: `**Solution**  '''sql /* 1️⃣  Use explicit JOIN syntax and LEFT JOIN to keep cust`

**summarize @ 1**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials used the sun’s shadow but were inaccura`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and striating bedrock`

**summarize @ 2**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials used the sun’s shadow but were inaccura`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and striating bedrock`

**summarize @ 4**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials used the sun’s shadow but were inaccura`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and striating bedrock`

**summarize @ 8**

- #0 summarize in=813 out=64 fin=length: `- **Early time‑keeping**:     * Sundials used the sun’s shadow but were inaccura`
- #1 summarize in=1434 out=64 fin=length: `**Summary**   Glaciers erode valleys by sliding, plucking, and striating bedrock`
