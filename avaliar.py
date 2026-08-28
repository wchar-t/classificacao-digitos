"""Parte 2: avaliação no conjunto de teste.

Calcula:
  - acurácia
  - matriz de confusão 10x10
  - par de dígitos mais confundidos
  - três imagens erradas (índice, rótulo verdadeiro, previsão)
Salva:
  - relatorio_dados.json com tudo acima
  - relatorio_erros.png com as três imagens erradas
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

modelo = keras.models.load_model("modelo.keras")
(_, _), (x_te, y_te) = keras.datasets.mnist.load_data()
x_te_n = (x_te / 255.0).astype("float32")[..., None]

test_loss, test_acc = modelo.evaluate(x_te_n, y_te, verbose=0)
print(f"Acurácia no teste: {test_acc:.4f}")
print(f"Loss no teste:     {test_loss:.4f}")

probs = modelo.predict(x_te_n, verbose=0)
y_pred = probs.argmax(axis=1)

cm = confusion_matrix(y_te, y_pred)
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(cm)

erros = cm.copy()
np.fill_diagonal(erros, 0)
flat = []
for i in range(10):
    for j in range(10):
        if i != j:
            flat.append((erros[i, j], i, j))
flat.sort(reverse=True)
top1 = flat[0]
top2 = flat[1] if len(flat) > 1 else None
print(f"\nMais confundidos: real {top1[1]} -> previsto {top1[2]} ({top1[0]} vezes)")
if top2:
    print(f"Segundo par:      real {top2[1]} -> previsto {top2[2]} ({top2[0]} vezes)")

err_mask = y_pred != y_te
err_idx = np.where(err_mask)[0]
print(f"\nTotal de erros: {len(err_idx)}")

rng = np.random.default_rng(0)
amostra = rng.choice(err_idx, size=min(3, len(err_idx)), replace=False)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, idx in zip(axes, amostra):
    ax.imshow(x_te[idx], cmap="gray")
    p_errado = probs[idx, y_pred[idx]]
    p_certo = probs[idx, y_te[idx]]
    ax.set_title(
        f"previsto {y_pred[idx]} ({p_errado*100:.1f}%)\nreal {y_te[idx]} ({p_certo*100:.1f}%)",
        fontsize=10,
    )
    ax.axis("off")
fig.suptitle("Três erros do modelo no conjunto de teste", fontsize=12)
fig.tight_layout()
fig.savefig("relatorio_erros.png", dpi=120)
print("Salvei relatorio_erros.png")

dados = {
    "test_acc": float(test_acc),
    "test_loss": float(test_loss),
    "confusion_matrix": cm.tolist(),
    "top_confusions": [
        {"real": int(c[1]), "previsto": int(c[2]), "vezes": int(c[0])} for c in flat[:5]
    ],
    "erros_amostra": [
        {
            "indice": int(idx),
            "real": int(y_te[idx]),
            "previsto": int(y_pred[idx]),
            "p_previsto": float(probs[idx, y_pred[idx]]),
            "p_real": float(probs[idx, y_te[idx]]),
        }
        for idx in amostra
    ],
    "total_erros": int(len(err_idx)),
}
with open("relatorio_dados.json", "w") as f:
    json.dump(dados, f, indent=2)
print("Salvei relatorio_dados.json")
