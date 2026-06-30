# **"Identification of Folk Music / Folk Music Classification"**
 
Music Information Retrieval (MIR) on Indian folk music — built as part of the
**IKS Internship Program**, project : *Analysis of Raag Origins of Indian Folk*


##  **1. Pre-processing Core Idea :**

Each `.pickle` file consists two very different things together — thousands of **mel-spectrograms** (the actual sound) and **metadata** (who sang it, what song, what style). Cramming both into one CSV means writing raw audio numbers as *text* — bloating a single file to several gigabytes for nothing.

So we split them:

| **mel-spectrograms** | **metadata** |
|---|---|
| `{style}_melspec.npy` | `{style}.csv` |
| Binary, compact, fast to load | song · genre · artist · gender |
| Shape `(N, 128, 130)` | One row per segment |


## Status
 
🚧 Work in progress — dataset and modeling notebooks are being added
iteratively as the internship research progresses.
 
## Acknowledgements
 
IKS Internship Program 2026, NIT Silchar.
