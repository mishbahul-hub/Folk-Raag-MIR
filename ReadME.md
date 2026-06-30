# **"Identification of Folk Music / Folk Music Classification"**
 
Music Information Retrieval (MIR) on Indian folk music — built as part of the
**IKS Internship Program**, project : *Analysis of Raag Origins of Indian Folk*

##  **1. Dataset Used : Indian Folk Music Dataset**

A mel-spectrogram dataset spanning **15 Indian folk styles**, built to capture musical traditions that are scarce online.

|  Stats | Value |
|---|---|
| Recordings | 606 |
| Total duration | 54.45 hrs |
| Folk styles | 15 |
| Artists | 125 (34 F · 91 M) |
| Recordings per style | 16 – 50 |
| Artists per style | 4 – 22 |

### Folk Styles

| | | | |
|---|---|---|---|
| Bauls | Bhavageethe | Garba | Kajri |
| Maand | Sohar | Tamang Selo | Veeragase |
| Bhatiali | Bihu | Gidha | Lavani |
| Naatupura Paatu | Sufi | Uttarakhandi | |

### Feature Extraction

Each recording is sliced into **3-second segments** with a **0.5-second sliding window**, then converted to a mel-spectrogram. Every segment is annotated with:

`folk_style` · `state` · `artist` · `gender` · `song` · `source` · `no_of_artists` · `folk_style_id` · `state_id` · `artist_id` · `gender_id`
##  **2. Pre-processing Core Idea :**

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
