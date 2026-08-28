# O que você vai construir

Uma página web que reconhece um dígito desenhado à mão, sem servidor e sem internet depois de carregada.

O modelo é treinado por você em Python, exportado, e roda dentro do navegador de quem abre a página.

## Parte 1 · Treinar

No Google Colab, treine uma rede convolucional no MNIST completo — 60.000 imagens de treino e 10.000 de teste.

```python
from tensorflow import keras

(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
x_tr = (x_tr / 255.0).astype('float32')[..., None]
x_te = (x_te / 255.0).astype('float32')[..., None]
```

### Requisitos do modelo

|                |                                                 |
| -------------- | ----------------------------------------------- |
| **entrada**    | 28 × 28 × 1                                     |
| **pelo menos** | duas camadas convolucionais                     |
| **saída**      | `Dense(10, activation='softmax')`               |
| **perda**      | `sparse_categorical_crossentropy`               |
| **callback**   | `EarlyStopping` com `restore_best_weights=True` |

A arquitetura é sua escolha. Você precisa justificá-la no relatório.

## Parte 2 · Avaliar antes de exportar

Não exporte um modelo que você não avaliou.

Reporte no relatório:

* Acurácia no conjunto de teste
* Matriz de confusão 10 × 10
* Os dois dígitos que o modelo mais confunde entre si
* Três imagens que ele errou, com a previsão e o rótulo verdadeiro

Errar 4 por 9 é diferente de errar 1 por 8. A matriz mostra qual erro o seu modelo comete, e a acurácia sozinha esconde isso.

## Parte 3 · Exportar

```bash
pip install tensorflowjs
```

```python
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(modelo, 'modelo_web')
```

Isso gera uma pasta com `model.json` e um ou mais arquivos `.bin`.

**Atenção.** O tensorflowjs costuma conflitar com a versão do TensorFlow instalada. Se quebrar, instale num ambiente separado ou converta a partir do arquivo salvo:

```bash
tensorflowjs_converter --input_format=keras modelo.keras modelo_web/
```

## Parte 4 · A página

Um arquivo HTML, sem framework, com:

* Um canvas onde se desenha com mouse e com o dedo
* Um botão para limpar
* O dígito previsto em destaque
* A confiança de cada uma das 10 classes, em barras

O esqueleto do JavaScript:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script>
let modelo;
async function carregar() {
  modelo = await tf.loadLayersModel('modelo_web/model.json');
}

function prever() {
  const t = tf.browser.fromPixels(canvas, 1)   // 1 canal
              .resizeBilinear([28, 28])
              .toFloat()
              .div(255.0)
              .expandDims(0);                  // vira lote de 1
  const p = modelo.predict(t).dataSync();
  // p tem 10 posicoes, uma por digito
}
</script>
```

## Parte 5 · Publicar

Suba no GitHub e ative o GitHub Pages. Entregue o link funcionando.

