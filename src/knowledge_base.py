DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "Supervised Learning",
        "source": "ML Fundamentals",
        "text": (
            "Supervised learning is a type of machine learning where a model is trained on "
            "labelled data. Each training example has an input and a known correct output. "
            "The model learns to map inputs to outputs by minimising a loss function over "
            "many examples. Common algorithms include linear regression for continuous "
            "outputs and logistic regression for classification. The key challenge is "
            "generalisation — the model must perform well on data it has never seen, not "
            "just on the training set. Overfitting occurs when the model learns the training "
            "data too closely and fails on new examples. Techniques like cross-validation, "
            "regularisation, and dropout help prevent this."
        ),
    },
    {
        "id": "doc_02",
        "title": "Neural Networks",
        "source": "Deep Learning Basics",
        "text": (
            "A neural network is a function approximator made up of layers of nodes called "
            "neurons. Each neuron takes a weighted sum of its inputs, adds a bias, and passes "
            "the result through an activation function such as ReLU or sigmoid. The network "
            "learns by adjusting weights using backpropagation — an algorithm that computes "
            "the gradient of the loss with respect to each weight using the chain rule. "
            "Stochastic gradient descent then updates the weights in the direction that "
            "reduces the loss. Deeper networks can learn more complex representations but "
            "require more data and are harder to train. Batch normalisation and skip "
            "connections were introduced to address these training difficulties."
        ),
    },
    {
        "id": "doc_03",
        "title": "Transformer Architecture",
        "source": "Attention Is All You Need",
        "text": (
            "The transformer architecture, introduced by Vaswani et al. in 2017, replaced "
            "recurrent networks for sequence modelling tasks. Its core mechanism is "
            "self-attention, which allows every position in a sequence to directly attend "
            "to every other position. This removes the sequential bottleneck of RNNs and "
            "enables parallelisation during training. The encoder maps an input sequence "
            "to a set of continuous representations, and the decoder generates the output "
            "sequence one token at a time. Positional encodings are added to the input "
            "embeddings to give the model information about token order, since self-attention "
            "itself is permutation-invariant."
        ),
    },
    {
        "id": "doc_04",
        "title": "Large Language Models",
        "source": "LLM Overview",
        "text": (
            "Large language models are transformer-based models trained on massive text "
            "corpora to predict the next token in a sequence. Scale has proven to be the "
            "primary driver of capability — models with more parameters trained on more data "
            "consistently outperform smaller ones on a wide range of tasks. GPT-4, LLaMA, "
            "and Gemini are examples of modern LLMs. These models are typically fine-tuned "
            "after pretraining using reinforcement learning from human feedback to align "
            "their outputs with human preferences. A known limitation is hallucination — "
            "LLMs sometimes generate confident-sounding but factually incorrect statements "
            "because they model token distributions rather than ground truth."
        ),
    },
    {
        "id": "doc_05",
        "title": "Retrieval-Augmented Generation",
        "source": "Lewis et al. 2020",
        "text": (
            "Retrieval-Augmented Generation, introduced by Lewis et al. in 2020, addresses "
            "the hallucination problem of LLMs by grounding answers in retrieved documents. "
            "At query time, the system retrieves the most relevant passages from an external "
            "knowledge base and passes them alongside the question to the LLM. The model is "
            "instructed to answer only from the provided context. This approach allows the "
            "knowledge base to be updated without retraining the model and provides "
            "traceable citations for each answer. The quality of a RAG system depends "
            "heavily on retrieval quality — if the wrong documents are retrieved, the LLM "
            "cannot produce a correct answer regardless of its capability."
        ),
    },
    {
        "id": "doc_06",
        "title": "Vector Embeddings",
        "source": "Representation Learning",
        "text": (
            "Vector embeddings are dense numerical representations of text that capture "
            "semantic meaning. Two sentences with similar meanings will have embeddings that "
            "are close to each other in vector space, even if they share no words. Embeddings "
            "are produced by encoder models trained on large corpora. In RAG systems, both "
            "the documents in the knowledge base and the user query are embedded, and "
            "retrieval is performed by finding the documents whose embeddings are closest "
            "to the query embedding. Cosine similarity and dot product are the most common "
            "distance metrics used. The quality of the embedding model directly affects "
            "retrieval quality and therefore the overall RAG pipeline performance."
        ),
    },
    {
        "id": "doc_07",
        "title": "Fine-Tuning vs RAG",
        "source": "Applied LLM Engineering",
        "text": (
            "Fine-tuning modifies the weights of a pretrained model on a task-specific "
            "dataset. It is effective for teaching the model a new style, format, or "
            "domain-specific terminology. However, it is expensive to repeat when knowledge "
            "changes, and the model can still hallucinate on facts it learned during "
            "fine-tuning. RAG keeps the base model frozen and instead supplies relevant "
            "context at inference time. This makes knowledge updates cheap — you only need "
            "to update the document store. For factual question answering over changing "
            "data, RAG is generally preferred. For tasks where the model needs to change "
            "its behaviour or output format, fine-tuning is more appropriate. The two "
            "approaches can also be combined."
        ),
    },
    {
        "id": "doc_08",
        "title": "Vector Databases",
        "source": "AI Infrastructure",
        "text": (
            "Vector databases are purpose-built storage systems for high-dimensional "
            "embeddings. Unlike relational databases that search by exact match, vector "
            "databases support approximate nearest-neighbour search, which finds the most "
            "similar embeddings to a query vector efficiently at scale. ChromaDB, Pinecone, "
            "Weaviate, and Qdrant are widely used options. ChromaDB is popular for local "
            "development and research because it runs in-process with no external service "
            "required. Pinecone is a managed cloud service used in production systems. The "
            "choice of vector database affects query latency, scalability, and cost but "
            "does not affect the quality of the embeddings themselves."
        ),
    },
    {
        "id": "doc_09",
        "title": "Chunking Strategy",
        "source": "RAG Engineering",
        "text": (
            "Chunking is the process of splitting documents into smaller pieces before "
            "embedding them. It matters because embedding an entire document produces a "
            "vector that represents the average meaning of the whole document, which is "
            "too broad to match a specific query accurately. Smaller chunks are more "
            "precise but may lack context. A common approach is to use fixed-size chunks "
            "of around 300 to 500 tokens with an overlap of 10 to 20 percent between "
            "consecutive chunks. The overlap ensures that information at chunk boundaries "
            "is not lost. Sentence-level chunking and semantic chunking are alternatives "
            "that try to split at natural content boundaries rather than fixed token counts."
        ),
    },
    {
        "id": "doc_10",
        "title": "RAG Evaluation",
        "source": "RAGAS Paper — Es et al. 2023",
        "text": (
            "Evaluating RAG systems is more complex than evaluating a standard classifier "
            "because quality depends on two separate components: the retriever and the "
            "generator. The RAGAS framework defines four core metrics. Faithfulness measures "
            "whether each claim in the generated answer is supported by the retrieved "
            "context. Answer relevancy measures how directly the answer addresses the "
            "question. Context precision measures the proportion of retrieved chunks that "
            "are actually relevant to the question. Context recall measures whether all "
            "the information needed to answer the question was present in the retrieved "
            "chunks. Taken together, these metrics give a diagnostic view of where a "
            "pipeline is failing."
        ),
    },
]
