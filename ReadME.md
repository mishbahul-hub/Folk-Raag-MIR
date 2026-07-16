# **"Identification of Folk Music / Folk Music Classification"**
 
Music Information Retrieval (MIR) on Indian folk music — built as part of the
**IKS Internship Program**, project : *Analysis of Raag Origins of Indian Folk*

#  **1. Dataset Used : Indian Folk Music Dataset**

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
#  **2. Pre-processing Core Idea :**

Each `.pickle` file consists two very different things together — thousands of **mel-spectrograms** (the actual sound) and **metadata** (who sang it, what song, what style). Cramming both into one CSV means writing raw audio numbers as *text* — bloating a single file to several gigabytes for nothing.

So we split them:

| **mel-spectrograms** | **metadata** |
|---|---|
| `{style}_melspec.npy` | `{style}.csv` |
| Binary, compact, fast to load | song · genre · artist · gender |
| Shape `(N, 128, 130)` | One row per segment |

Three preprocessing steps are applied to every mel-spectrogram:

1. **Length standardization (`fix_length`)** — forces every spectrogram to the exact same shape (128 × 130) by center-cropping if too long, or zero-padding if too short.
2. **Normalization (`standardize`)** — rescales each spectrogram to zero mean and unit standard deviation.
3. **SpecAugment-style augmentation (`spec_augment`)** — randomly blacks out small frequency/time chunks during training to make the model more robust.

   <img width="344" height="172" alt="kaggle_data_explorer" src="https://github.com/user-attachments/assets/3305d61a-9621-4e21-a009-b733aa47e04b" />


# **3. Related Works :**
| Paper Title | Year | Dataset | Model Used | Accuracy |
|---|---|---|---|---|
| An improved ViT model for music genre classification based on mel spectrogram (Wu et al.) | 2025 | GTZAN (10 genres) | Improved ViT4 (CNN + Transformer hybrid + Enhanced Channel Attention) | 86.6% |
| Music Genre Classification Using Prosodic, Stylistic, Syntactic and Sentiment-Based Features (Kovács & Baghiu) | 2025 | 14,000 Romanian songs (tabulaturi.ro), 7 merged genres | Logistic Regression | 69.94% mean F1 (test set) |
| uFoC: Regional Classification of Turkish Folk Music | 2025 | TuFoC — 369 recordings, 5 Turkish regions | CNN (MobileNetV2, transfer learning) | 92.68% |
| Musical Genre Classification Using Mel-Spectrograms and Mel-Scalograms | 2025 | GTZAN (10 genres) | MobileNetV2 / EfficientNetB0 / ResNet50 / VGG16 | 76.7±4.9% / 83.5±2.2% / 83.3±1.5% / 74.3±3.6% |

# **4. Data Visualization :**

**Sample counts per folk style** — class sizes range from under 10,000 (Naatupura_Paatu) to nearly 40,000 (Sufi), reflecting real-world imbalance in how well each style is digitized/represented:

<img width="936" height="334" alt="sample_counts_per_style" src="https://github.com/user-attachments/assets/678d9aaa-680b-4ac8-a2c6-70fd4cf3474a" />


**Mel-spectrogram examples** (3 of 15 classes) — note how distinct the spectral texture is even visually between styles:

<img width="1153" height="368" alt="melspectrogram_examples" src="https://github.com/user-attachments/assets/049cb8e9-179e-446c-9ead-6dd20d715b21" />

# **5. Results :**
All scores below are **macro-averaged** across the 15 folk-style classes.

| Model | Val. Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| AlexNet | 0.730 | 0.70 | 0.69 | 0.69 |
| ResNet-34 | 0.750 | 0.71 | 0.70 | 0.70 |
| **ResNet-18** | **0.762** | **0.72** | **0.72** | **0.71** |
| MobileNet | 0.750 | 0.71 | 0.70 | 0.70 |
| EfficientNet-B0 *(SOTA)* | 0.760 | 0.73 | 0.72 | 0.71 |
| AST *(SOTA, pending)* | 0.71 | — | — | — |

# **6. Confusion Matrix :**

**EfficientNet-B0** — validation set:

<img width="685" height="655" alt="confusion_matrix_efficientnet" src="https://github.com/user-attachments/assets/dd9034ee-004d-4a14-bb48-d5a9d920e631" />

# **7. Status :**
🚧 Complete AST fine-tuning and add it to the results comparison table.
🚧 Explore **PaSST (2022)** as an improved AST variant.

 
# **8. Acknowledgements :**
 
*IKS Internship Program 2026, NIT Silchar.*
