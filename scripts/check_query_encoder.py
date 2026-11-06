from retrieval.query_encoder import QueryEncoder


encoder = QueryEncoder()

query = "How are neural networks used in medical image classification?"

embedding = encoder.encode_query(query)

print("Query:", query)
print("Embedding dimensions:", len(embedding))
print("First five values:", embedding[:5])