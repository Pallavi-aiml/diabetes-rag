
from transformers import pipeline

model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    top_k=None,
)

evidence = (
    "The DPP demonstrated that intensive lifestyle intervention "
    "could reduce the risk of incident type 2 diabetes by 58% over 3 years."
)

claim = (
    "Intensive lifestyle intervention can reduce the risk of "
    "incident type 2 diabetes by 58% over a 3-year period."
)

print("LABEL MAP:", model.model.config.id2label)

print("\nTEST 1: Exact evidence")
print(model({"text": evidence, "text_pair": evidence}, truncation=True))

print("\nTEST 2: Paraphrased claim")
print(model({"text": evidence, "text_pair": claim}, truncation=True))

