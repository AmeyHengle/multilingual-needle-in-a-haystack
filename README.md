
# Multilingual Needle-in-a-Haystack (MLNeedle). 

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

## Multilingual Question Answering.

Our proposed MLNeedle test is an extension of the multilingual QA task to the long-context format. For this, we choose the MLQA dataset because of it's parallel data structure. That is, for a given question, MLQA provides the relevant information ( `needle` ) in multiple languages. As example, please consider the following instance from the MLQA dataset which highlights it's parallel structure. 

| ID                                   | Question                     | Context (Needle)                                                                                                                                                                     | Question Lang | Context Lang | Answer (Groundtruth)                                                                                             |
|--------------------------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|--------------|------------------------------------------------------------------------------------------------------------------|
| 1 | How can you activate EFS?    | The support of EFS is not available in Basic, Home, and MediaCenter versions of Windows, and must be activated after installation of Professional, Ultimate, and Server versions of Windows or by using enterprise deployment tools within Windows domains. | en            | en           | after installation of Professional, Ultimate, and Server versions of Windows or by using enterprise deployment tools within Windows domains |
| 1 | How can you activate EFS?    | EFS की सुविधा Windows के बेसिक, होम तथा मीडिया सेंटर संस्करण में उपलब्ध नहीं है तथा Windows के प्रोफेशनल, अल्टीमेट तथा सर्वर संस्करण के लोड करने के पश्चात या Windows डोमेन में इंटरप्राईज़ द्वारा प्रदत्त टूल का प्रयोग करके एक्टिवेट की जानी चाहिए।                  | en            | hi           | Windows के प्रोफेशनल, अल्टीमेट तथा सर्वर संस्करण के लोड करने के पश्चात या Windows डोमेन में इंटरप्राईज़ द्वारा प्रदत्त टूल का प्रयोग करके                 |
| 1 | How can you activate EFS?    | Sự hỗ trợ của EFS không có sẵn trong các phiên bản Basic, Home và MediaCenter của Windows, và nó phải được kích hoạt sau khi cài đặt các phiên bản Professional, Ultimate và Server của Windows hay bằng cách sử dụng các công cụ đặc biệt.                  | en            | vi           | sau khi cài đặt các phiên bản Professional, Ultimate và Server của Windows hay bằng cách sử dụng các công cụ đặc biệt.           |
| 1 | How can you activate EFS?    | Basic、Home和MediaCenter版本的Windows不支持磁盘限额功能。要使用这个功能，必须安装Professional、Ultimate或者服务器版本的Windows，或者使用Windows域中的企业部署工具进行部署。                | en            | zh           | 安装Professional、Ultimate或者服务器版本的Windows，或者使用Windows域中的企业部署工具进行部署。                                          |



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
