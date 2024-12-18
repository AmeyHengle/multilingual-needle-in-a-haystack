### Reviewer 1

> **I don't think the "lost in the middle" phenomenon is prominent in Figure 4. Either the position could be more fine-grained to verify the trend in the accuracy, or the descriptions should be more precise (Line 301-303)**

1. We appreciate the reviewer’s observation regarding the granularity of needle positions in Figure 4 and its impact on illustrating the "lost in the middle" phenomenon.
3. In the revised manuscript, we commit to updating Figure 4 with results from a finer-grained analysis of needle positions. Specifically, we will expand the current evaluation (start \(k=0\), middle \(k=n/2\), and end \(k=n\)) by including six additional intermediate positions across the context. This updated figure will provide a more detailed view of the trend in accuracy, further verifying the "lost in the middle" phenomenon and offering a clearer understanding of how needle position affects performance.
4. Having this benchmark will enable the community 

---

> **Why separate the results in Figure 4 into high-medium/low-resource languages? The trend looks similar.**

  1.  Our intention behind separating the results into high-, medium-, and low-resource language groups was mainly to emphasize that the "lost in the middle" phenomenon is consistent across languages, irrespective of their resource availability during pretraining. By doing so, we establish that this issue is likely a consequence of a **positional attention bias**, rather than being language- or resource-specific. 
2. In response to the reviewer's feedback, we propose to including a more detailed version of Figure 4 in the Appendix of the revised manuscript. This new figure will display the trends for individual languages without grouping, providing additional granularity for readers who wish to analyze the results at the language level.

---


> **I don't think zh, hi, vi, ar should be categorized as low-resource languages. See discussions in [1]** 
1. We would like to note that the languages were categorized as high-, medium-, or low-resource based on their **% representation in the pretraining corpus**. Specifically, languages such as Vietnamese (vi), Simplified Chinese (zh), and Hindi (hi) were categorized as low-resource because they account for only 0.08%, 0.13%, and 0.01% of the Llama-2 pretraining corpus, respectively [2]. 
2. We appreciate the reviewer's reference to [1] and acknowledge that our way of categorizing langauges into high/medium/low resource may not fully align with discussions in contemporary literature. 
5. Taking the reviewer’s feedback into account, we propose an alternative approach for structuring the results in Figure 4:

	i) **Grouping Based on Similarity to English**  
	    We propose to group languages according to their **syntactic and phonetic similarity to English**. This allows us to explore whether linguistic distance plays  role in the observed performance trends.
	    
	  ii) **Quantifying Similarity**  
	    To define similarity, we use **lang2vec** [3], which provides vector-similarity functions to compute syntactic and phonetic distance between languages. Using this method, we find the following grouping:

	| Language | Syntactic Distance | Phonological Distance | Group |
	|----------|---------------------|------------------------|-------|
	| eng       | 0                   | 0.0002                | 1     |
	| deu      | 0.42                | 0.3277                | 2     |
	| spa      | 0.4                 | 0.3433                | 2     |
	| hin      | 0.59                | 0.3433                | 3     |
	| vie      | 0.57                | 0.427                 | 3     |
	| zho      | 0.57                | 0.5687                | 3     |
	| ara      | 0.57                | 0.5687                | 3     |
   	    
  -   **Group 1**: English (eng)
  -   **Group 2**: German (deu) and Spanish (spa)
  -   **Group 3**: Hindi (hin), Vietnamese (vie), Simplified Chinese (zho), and Arabic (ara)

   iii) **Revised Figure 4**  
    The revised version of Figure 4 will reflect this new grouping, which we believe offers a linguistically meaningful way to interpret the results. We commit to include this table in the Appendix and further detail our rationale and the use of lang2vec in the revised manuscript .



#### References
[1] The State and Fate of Linguistic Diversity and Inclusion in the NLP World. https://aclanthology.org/2020.acl-main.560 <br>
[2]Hugo Touvron and Louis Martin. 2023. Llama 2: Open foundation and fine-tuned chat models. https://arxiv.org/abs/2307.09288 <br>
[3] URIEL and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors https://aclanthology.org/2024.tacl-1.9/

---
---

### Reviewer 2
> **I think the limitation regarding the dependency on the MLQA dataset is a potential concern, as it makes it difficult to differentiate whether LLMs are utilizing contextual understanding or simply relying on curated knowledge. A wider variety of needle sources could enhance the robustness of the findings.**

1.  While the limitation regarding MLQA dataset is valid, our experiments and ablation study may provide sufficient evidence to conjecture that LLMs are not answering from curated / parametric knowledge.
2.  Firstly, our experiments **(Figure 4)** prove that LLM performance is highly sensitive to the position of the needle within the context. Specifically, models exhibit a "lost in the middle" phenomenon, struggling to retrieve the same piece of information when it is located in the middle of a long input. This positional bias would not be evident if models were solely relying on parametric retrieval, as both the query and the needle remain unchanged regardless of position. This suggests that the LLMs are processing input context dynamically. 
3.  This is also supported by results in **Table 1 (Baseline)**. We can see that for all LLMs, the baseline performance (i.e with no distractor passages) is not-perfect.   
4. As further evidence, our ablation study using existence accuracy (Figure 7) shows a similar trend. Existence accuracy is consistently affected by both the language and the position of the needle. If models were relying purely on parametric knowledge, they would achieve perfect existence accuracy (a score of 1) regardless of these variables. Instead, we observe performance variations, further supporting that LLMs are leveraging the input-context rather than memorized knowledge.

---


> **The paper could benefit from a more detailed discussion of the constraints experienced with the selected models, especially regarding those not evaluated, like larger parameter LLMs (e.g., Command-R or GPT-4). Understanding the potential improvements from these models would give readers a more complete view of the challenges in multilingual contexts.**
1. We could not include Command-R and GPT-4 due to budget constraints. However, we make our evaluation as extensive as possible - using a diverse setup of LLMs like Mistral, Cohere-Aya, and Llama3. Furthermore, we also conduct ablation studies with varying decoding strategies and compare base vs instruction-tuned models. 
2. We also acknowledge the reviewer's suggestion. A detailed discussion regarding the constraints experienced in selecting the evaluated models and the potential impact of including larger parameter LLMs could benifit our study.
3. In response, we commit to include a detailed explanation in **section 2.3 (Models)** of the revised manuscript to clarify the reasons for our model selection, including limitations related to computational resources, budget, and access to proprietary models. 
4. Additionally, we will also commit to expanding **section 7 (Limitations)** to discuss the potential benefits and challenges of evaluating larger models in multilingual and long-context settings.

---


> **I find that the experimentation could have included more languages, particularly focusing on those less represented in the study, to better gauge the model's performance across varied linguistic landscapes.**
1. Our experiments are based on the MLQA dataset, which forms the foundation for our MLNeedle test framework. Consequently, our language selection was limited to those included in MLQA. 
2. We choose to rely purely on existing open-source datasets like MLQA and mMarco to ensure that all languages have high-quality, aligned question-answer pairs and add reliability to our experiements -- following the precedence of a similar study in English long-context evaluation [4].
3.  Furthermore, we believe that the seven languages included in MLQA—English (eng), Arabic (ara), German (deu), Spanish (spa), Hindi (hin), Vietnamese (vie), and Simplified Chinese (zho)— represent a wide range of linguistic diversity. These languages span:
	-   **Language families**: Indo-European (eng, spa, hin, deu), Afro-Asiatic (ara), Sino-Tibetan (zho), and Austroasiatic (vie).
	-   **Scripts**: Latin (eng, spa, deu, vie), Arabic (ara), Devanagari (hin), and Simplified Chinese (zho).
	-   **Regions**: Languages from Europe, the Americas, South Asia, East Asia, and the Middle East are included.
4.  Hence, even though adding more languages to our study would add confidence to our findings, we feel that the results and their implications might not necessarily change. Given the diversity of the chosen languages, we expect that our findings—such as sensitivity to needle position — would potentially generalise to other languages not currently included in the study.
5. Lastly, we would also like to highlight that our MLNeedle test framework is designed to be modular and flexible. This will allow us to expand it iteratively. In future work, we intend to incorporating additional languages, particularly low-resoured and regional ones. 

---

> **The experimental design discusses significant performance differences but would be more impactful if it included qualitative examples of model outputs across the diverse languages and contexts tested, as this would offer clearer insights into where current models struggle or succeed.**
1.  We acknowledge the reviewer's suggestion and agree that qualitative examples can offer valuable insights into model performance.
2. While our manuscript already includes some dataset-specific examples in Figures 2 and 9, we commit to adding a similar table of qualitative examples (model-outputs and error cases) in Appendix F of our revised manuscript.


#### References
[4] Lost in the Middle: How Language Models Use Long Contexts (Liu et al., TACL 2024) https://aclanthology.org/2024.tacl-1.9/

---
---


### Reviewer 3


> **The paper does not present any novel techniques, but I think that's appropriate for this kind of paper.**

1. We appreciate the reviewer’s understanding that introducing novel techniques is not the primary goal of this work. 
2. Grounded in the principle ***"what cannot be measured, cannot be improved"***, the primary motivation of our study is to establish a reliable and systematic framework for evaluating the long-context capabilities of multilingual LLMs.
3. While existing long-context evaluation benchmarks focus exclusively on English, to the best of our knowledge, our proposed MLNeedle test is the first framework designed to systematically evaluate and study the long-context behavior of LLMs in multilingual settings.
4. Hence, we believe that our benchmark will be useful for the community to 
---


> **My one major concern is about the exact accuracy metric as it applies to non-English text. Non-English text here is at a disadvantage-- it has to be translated into English first, before it can be compared against the answer. This has to be a lossy process. Have you attempted to measure the error rate of this process? One could imagine that if translation were poor, it could account for the degradation you are currently attributing to the language of the relevant information and the LLM's ability to retrieve from it.**
1. We acknowledge that translating non-English model outputs into English for comparison introduces the potential for translation errors, which could affect the exact accuracy scores. While we have not explicitly measured translation error rates in this study, we took some steps to minimize their potential impact.
2. To begin with, we use the **Google Translate service**, which is recognized for its high success rates in multilingual translation, particularly for widely spoken languages such as those included in our study [5].
3. Furthermore, we chose MLQA as the source for the "needles" specifically because of its highly parallel structure:
    - **MLQA provides ground-truth answers in at least four languages for each example**. This ensures that exact accuracy computations are more reliable compared to datasets with fewer or less aligned multilingual ground-truths.
    - **The dataset is aligned at the sentence level, minimizing ambiguity in ground-truth answers during translation**. This helps achieving consistency when evaluating model outputs across different languages. 
4. We appreciate the reviewer’s suggestion and will include a discussion of this potential limitation in the revised manuscript. 

---


> **I am also a little confused about the existence accuracy metric. The way you have constructed the context, isn't the information always present? Or do you omit it some percentage of the time?**
1. **Existence accuracy** measures whether a model can correctly determine if the relevant information (the "needle") is present within the input context. This is different from **exact accuracy**, which evaluates whether the model retrieves and outputs the correct answer explicitly. Existence accuracy serves as a simpler, yet critical, task for evaluating a model's **ability to locate relevant information in a multilingual, long-context** setting.
2. The reviewer is correct in noting that the needle is always included in the input context. We do not omit it. **In essense, existence accuracy measures the recall value of the LLM.**
3. Ideally, this should mean that an LLM is always able to identify its presence. However, as discussed in **Section 4 (Line 400)**, our findings indicate that the ability of LLMs to recognize explicitly stated information is affected by both the language of the needle and its position within the context. This underscores the limitations of current LLMs in reliably handling multilingual and long-context inputs.
4. We commit to define and describe existence accuracy in a more clear manner in our final manuscript. 



#### References
[5] How accurate is Google Translate? Google Translate vs DeepL https://lokalise.com/blog/google-translate-accuracy/
