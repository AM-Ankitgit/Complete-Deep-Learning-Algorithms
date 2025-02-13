from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

reference = "RAG is a method that improves LLMs using retrieved data."
generated = "RAG helps LLMs by retrieving relevant information."

rouge = Rouge()
scores = rouge.get_scores(generated, reference)
bleu = sentence_bleu([reference.split()], generated.split())

print("ROUGE Score:", scores)
print("BLEU Score:", bleu)
