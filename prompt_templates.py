prompt_template1 = lambda question, context: f"""
Answer the given question correctly based on the given context.

Context: {context}

Question: {question}

Answer:
""".strip()