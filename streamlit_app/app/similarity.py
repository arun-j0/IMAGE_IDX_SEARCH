from sklearn.metrics.pairwise import cosine_similarity
import torch

def calculate_similarity(embedding1, embedding2):
    if isinstance(embedding1, list):
        embedding1 = torch.tensor(embedding1)
    if isinstance(embedding2, list):
        embedding2 = torch.tensor(embedding2)

    embedding1 = embedding1.cpu().numpy()
    embedding2 = embedding2.cpu().numpy()

    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]
