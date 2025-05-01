from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

reference = "RAG is a method that improves LLMs using retrieved data."
generated = "RAG helps LLMs by retrieving relevant information."

rouge = Rouge()
scores = rouge.get_scores(generated, reference)
bleu = sentence_bleu([reference.split()], generated.split())

print("ROUGE Score:", scores)
print("BLEU Score:", bleu)






from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

rouge1 = Rouge()

scores = rouge1.get_scores(generated,reference)
print((scores))

sentence_bleu = sentence_bleu([generated.split()],reference.split())
print(sentence_bleu)

