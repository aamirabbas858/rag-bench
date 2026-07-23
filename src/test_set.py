TEST_CASES = [
    {
        "id": "q01",
        "question": "What is the difference between overfitting and generalisation in supervised learning?",
        "ground_truth": "Overfitting happens when a model learns the training data too closely and performs poorly on new unseen data. Generalisation refers to a model performing well on data it was not trained on, which is the actual goal of supervised learning.",
    },
    {
        "id": "q02",
        "question": "How does backpropagation train a neural network?",
        "ground_truth": "Backpropagation computes the gradient of the loss with respect to each weight using the chain rule, then stochastic gradient descent updates the weights in the direction that reduces the loss.",
    },
    {
        "id": "q03",
        "question": "Why was the transformer architecture significant compared to recurrent networks?",
        "ground_truth": "Transformers use self-attention which allows every position to attend to every other position directly, removing the sequential bottleneck of RNNs and enabling parallel training.",
    },
    {
        "id": "q04",
        "question": "What is hallucination in large language models?",
        "ground_truth": "Hallucination is when an LLM generates confident-sounding but factually incorrect statements because it models token distributions rather than ground truth.",
    },
    {
        "id": "q05",
        "question": "How does RAG reduce hallucination in language models?",
        "ground_truth": "RAG retrieves relevant documents and passes them alongside the question to the LLM. The model is instructed to answer only from the provided context, grounding the answer in real sources rather than memorised patterns.",
    },
    {
        "id": "q06",
        "question": "What makes two text embeddings similar to each other?",
        "ground_truth": "Two embeddings are similar when the texts they represent have similar meanings. Semantic similarity is captured as closeness in vector space, measured by cosine similarity or dot product.",
    },
    {
        "id": "q07",
        "question": "When should you use fine-tuning instead of RAG?",
        "ground_truth": "Fine-tuning is more appropriate when you need the model to change its behaviour, output format, or learn domain-specific terminology, rather than for factual question answering over changing data.",
    },
    {
        "id": "q08",
        "question": "What is the difference between ChromaDB and Pinecone?",
        "ground_truth": "ChromaDB runs in-process locally with no external service, making it suitable for development and research. Pinecone is a managed cloud service used in production systems.",
    },
    {
        "id": "q09",
        "question": "Why do RAG systems use chunking instead of embedding whole documents?",
        "ground_truth": "Embedding a whole document produces a vector that represents the average meaning of all its content, which is too broad to match a specific query. Smaller chunks produce more precise embeddings that match specific questions more accurately.",
    },
    {
        "id": "q10",
        "question": "What does context precision measure in RAG evaluation?",
        "ground_truth": "Context precision measures the proportion of retrieved chunks that are actually relevant to the question.",
    },
    {
        "id": "q11",
        "question": "What is the purpose of positional encodings in a transformer?",
        "ground_truth": "Positional encodings give the model information about the order of tokens in a sequence, since self-attention is permutation-invariant and cannot determine order on its own.",
    },
    {
        "id": "q12",
        "question": "What training objective do large language models use during pretraining?",
        "ground_truth": "Large language models are pretrained to predict the next token in a sequence.",
    },
    {
        "id": "q13",
        "question": "What is reinforcement learning from human feedback and why is it used?",
        "ground_truth": "RLHF is a fine-tuning technique used after pretraining to align LLM outputs with human preferences by using human feedback as a reward signal.",
    },
    {
        "id": "q14",
        "question": "What is context recall in RAG evaluation?",
        "ground_truth": "Context recall measures whether all the information needed to answer the question was present in the retrieved chunks.",
    },
    {
        "id": "q15",
        "question": "How does chunk overlap help in RAG systems?",
        "ground_truth": "Chunk overlap ensures that information at chunk boundaries is not lost when a document is split, because a portion of each chunk is repeated at the start of the next.",
    },
    {
        "id": "q16",
        "question": "What is faithfulness as a RAG evaluation metric?",
        "ground_truth": "Faithfulness measures whether each claim in the generated answer is supported by the retrieved context rather than being made up by the model.",
    },
    {
        "id": "q17",
        "question": "What is the role of the encoder in a transformer model?",
        "ground_truth": "The encoder maps an input sequence to a set of continuous representations that capture the meaning of the input.",
    },
    {
        "id": "q18",
        "question": "Why is RAG preferred over fine-tuning for factual question answering over changing data?",
        "ground_truth": "With RAG, updating knowledge only requires updating the document store. Fine-tuning would require retraining the model each time the knowledge changes, which is expensive.",
    },
    {
        "id": "q19",
        "question": "What activation functions are used in neural networks?",
        "ground_truth": "Common activation functions include ReLU and sigmoid.",
    },
    {
        "id": "q20",
        "question": "What does answer relevancy measure in RAG evaluation?",
        "ground_truth": "Answer relevancy measures how directly and completely the generated answer addresses the question that was asked.",
    },
]
