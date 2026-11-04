from ingestion.embedding_generator import EmbeddingGenerator


generator = EmbeddingGenerator()

texts = [
    "Artificial intelligence is transforming healthcare.",
    "Deep learning is used for medical image classification.",
    "The weather is pleasant today."
]

embeddings = generator.embed_texts(texts)

for index, embedding in enumerate(embeddings):

    print(f"\nText {index + 1}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First five values: {embedding[:5]}")