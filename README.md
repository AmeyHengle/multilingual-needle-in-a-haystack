
# Project Title
Multilingual Needle-in-a-Haystack (MLNeedle). 


### TLDR:

We introduce **MultiLingual Needle-in-a-Haystack (MLNeedle)** test, designed to evaluate the ability of multilingual large language models (LLMs) to retrieve relevant information from long multilingual contexts. Our results reveal that LLMs struggle with retrieving information when it is in non-Latin languages or positioned in the middle of a long context. Going further into these findings, we observe that the models show relatively stable performance when the distractor passages are in different languages but still face significant challenges in handling long multilingual contexts effectively.


## Installation

### Via `requirements.txt` (using `pip`)
To install the required dependencies using pip, you can run the following command:

```bash
pip install -r requirements.txt
```

### Via `environment.yml` (using `conda`)
To create a conda environment with the required dependencies, use the following command:

```bash
conda env create -f environment.yml
```

## Overview of Directory Structure

- **experiments/**: Contains the scripts and configuration files for running the experiments.
- **results_table/**: Includes the summary tables of results generated from the experiments.
- **runs/**: Stores the outputs and logs from the various experimental runs.

### Results Overview:

1. **Model Performance vs. Context Size**:
   - All models exhibit a significant drop in performance as the context length increases, indicating their limited capability to handle long contexts effectively.
   - **Table 1** provides a summary of each model's performance across context sizes (4K to 32K tokens). The effective context length is often much shorter than the claimed length by the model.

2. **Effect of Needle Position**:
   - The models perform best when the relevant information (needle) is positioned at the start or end of the input context.
   - A significant performance drop is observed when the needle is located in the middle, highlighting the "lost-in-the-middle" phenomenon.

3. **Effect of Needle Language**:
   - Performance is highest when the needle is in English or a language close to English (e.g., German or Spanish).
   - **Figure 3** illustrates the substantial drop in performance when the needle is in non-Latin languages such as Chinese or Arabic.

4. **Effect of Haystack Language**:
   - Changing the language of distractor documents (haystack) has a minimal impact on model performance, indicating that the models can prioritize relevant information effectively regardless of distractor language.
   - **Table 2** summarizes the pairwise accuracy of models when the language of the needle and haystack is varied.

5. **Ablation Studies**:
   - **Temperature Sampling vs. Greedy Decoding**: Both generation strategies yield comparable results across different context sizes.
   - **Instruction Fine-tuning**: Instruction-tuned models consistently outperform their base variants, particularly in multilingual and long-context scenarios.

6. **Statistical Significance**:
   - The accuracy stabilizes after approximately 2,500 samples, confirming the reliability of the evaluation outcomes.

7. **Visualization**:
   - **Figure 1** and **Figure 4** provide visual insights into the performance across different languages and needle positions.

## Citation
If you use this code or data in your work, please cite the original paper.
