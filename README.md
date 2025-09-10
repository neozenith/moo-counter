# Moo Counter 🐮

Simulation counter of possible moos achievable in <https://find-a-moo.kleeut.com/>

I wanted to simulate for a given puzzle:
- What is the theoretical maximum?
- What is the median?

**USAGE:**

```sh
usage: moo_counter.py [-h] [--puzzle PUZZLE] [--iterations ITERATIONS]

Moo Counter

options:
  -h, --help            show this help message and exit
  --puzzle PUZZLE       Path to the puzzle file
  --iterations ITERATIONS
                        Number of iterations for random sampling
```

Eg:

```sh
uv run src/moo_counter/moo_counter.py --puzzle 20250910-puzzle.moo
```

Example output:

```sh
Moo count 103:  3
Moo count 104:  5
Moo count 105:  42
Moo count 106:  309
Moo count 107:  1300
Moo count 108:  4581
Moo count 109: 🐮 13618
Moo count 110: 🐮🐮🐮 34884
Moo count 111: 🐮🐮🐮🐮🐮🐮 76016
Moo count 112: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 143040
Moo count 113: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 231502
Moo count 114: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 328803
Moo count 115: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 402089
Moo count 116: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 435431
Moo count 117: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 410109
Moo count 118: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 341110
Moo count 119: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 249493
Moo count 120: 🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮 159704
Moo count 121: 🐮🐮🐮🐮🐮🐮🐮🐮 90640
Moo count 122: 🐮🐮🐮🐮 45582
Moo count 123: 🐮 20193
Moo count 124:  7705
Moo count 125:  2717
Moo count 126:  868
Moo count 127:  199
Moo count 128:  42
Moo count 129:  11
Moo count 130:  2
Moo count 131:  1
Moo count 132:  1
Total valid 'moo' moves found: 172
Theoretical maximum moo count: 132 after 3000000 iterations
Total time taken: 135.74s
```
