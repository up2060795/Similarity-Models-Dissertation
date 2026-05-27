import matplotlib.pyplot as plt
import numpy as np

models = ["Product2Vec", "Sentence\nTransformer", "Fine-Tuned\nTransformer"]

precision_at_k = {
    "P@3": [0.223, 0.011, 0.016],
    "P@5": [0.190, 0.008, 0.011],
    "P@10": [0.148, 0.007, 0.007],
}

category_precision_at_k = {
    "Cat P@3": [0.725, 0.868, 0.863],
    "Cat P@5": [0.704, 0.859, 0.853],
    "Cat P@10": [0.672, 0.843, 0.837],
}

x = np.arange(len(models))
width = 0.25

# Behavioural Precision@K graph
fig, ax = plt.subplots(figsize=(8, 5))

for i, (metric, values) in enumerate(precision_at_k.items()):
    ax.bar(x + (i - 1) * width, values, width, label=metric)

ax.set_ylabel("Precision@K")
ax.set_title("Behavioural Precision@K Across Embedding Models")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 0.25)

plt.tight_layout()
plt.savefig("precision_at_k_bar_chart.png", dpi=300)
plt.show()


# Category Precision@K graph
fig, ax = plt.subplots(figsize=(8, 5))

for i, (metric, values) in enumerate(category_precision_at_k.items()):
    ax.bar(x + (i - 1) * width, values, width, label=metric)

ax.set_ylabel("Category Precision@K")
ax.set_title("Category-Level Precision@K Across Embedding Models")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("category_precision_bar_chart.png", dpi=300)
plt.show()
