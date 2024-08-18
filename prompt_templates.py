prompt_template_exact_accuracy = (
    lambda question, context: f"""
Write a high-quality answer for the given question using only the provided passages (some of which might be irrelevant).

### Passages
{context}

Question: {question}

Answer:
""".strip()
)

prompt_template_existence_accuracy = (
    lambda question, context: f"""
Read the following list of passages and indicate whether any of the passages contain the right answer for the given question. Format your output strictly as 'Yes' or 'No'.

### Passages
{context}

Question: {question}

Answer [Yes/No]:
""".strip()
)

prompt_template_ICE = lambda context_passages: "\n".join(
    [f"[{i}] {passage}" for i, passage in enumerate(context_passages)]
)
