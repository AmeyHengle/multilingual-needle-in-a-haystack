prompt_template1 = (
    lambda question, context: f"""
Answer the given question correctly based on the given context.

Context: {context}

Question: {question}

Answer:
""".strip()
)

prompt_template2 = (
    lambda question, context: f"""
Write a high-quality answer for the given question using only the provided passages (some of which might be irrelevant). The output must strictly be in a JSON format as follows - {{'answer': ''}}

### Passages
{context}

Question: {question}

Answer:
""".strip()
)

prompt_template2 = (
    lambda question, context: f"""
Write a high-quality answer for the given question using only the provided passages (some of which might be irrelevant). The output must strictly be in a JSON format as follows - {{'answer': ''}}

### Passages
{context}

Question: {question}

Answer:
""".strip()
)

prompt_template3 = (
    lambda question, context: f"""
Write a high-quality answer for the given question using only the provided passages (some of which might be irrelevant).

### Passages
{context}

Question: {question}

Answer:

The output must strictly be in a JSON format as follows - {{'answer': ''}}
""".strip()
)

prompt_template4 = (
    lambda question, context: f"""
Write a high-quality answer for the given question using only the provided passages (some of which might be irrelevant).
The output must strictly be in a JSON format as follows - {{'answer': ''}}

### Passages
{context}

Question: {question}

Answer:
""".strip()
)

ICE_template1 = lambda context_passages: " ".join(
    [passage for passage in context_passages]
)

ICE_template2 = lambda context_passages: "\n".join(
    [f"[{i}] {passage}" for i, passage in enumerate(context_passages)]
)

# def in_context_examples_template1(context_passages: list)->str:
#     return " ".join([passage for passage in context_passages])

# def in_context_examples_template2(context_passages: list)->str:
#     return "\n".join([f"[{i}] {passage}" for i,passage in enumerate(context_passages)])
