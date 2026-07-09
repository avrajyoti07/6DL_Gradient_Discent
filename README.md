# Gradient Descent — From Scratch (Insurance Purchase Prediction)

🔗 **Repo:** [github.com/avrajyoti07/6DL_Gradient_Discent](https://github.com/avrajyoti07/6DL_Gradient_Discent)

A logistic regression classifier — predicting whether someone buys insurance based on **age** and **affordability** — built two ways: once with `keras`, and once by hand-deriving and coding the **gradient descent** update rule from scratch. This is the sixth project in my Deep Learning starter phase, and the one that ties everything together: activation functions, matrix operations, and loss functions all combine here into one real, working training loop.

## What's in this repo

| File | Description |
|---|---|
| `5_Gradient_Descent.ipynb` | Main notebook — Keras baseline model + manual gradient descent implementation |
| `deep_insurance.txt` | Dataset (`age`, `affordability`, `bought_insurance`) — 30 records, CSV-formatted |
| `README.md` | You're here |

## Dataset

30 records with:

| Column | Description |
|---|---|
| `age` | Person's age |
| `affordability` | A normalized affordability score (0–1) |
| `bought_insurance` | Target — `1` if they bought insurance, `0` if not |

## What the notebook does

1. **Split the data** — `train_test_split` (80/20, `random_state=25`) → 24 training rows, 6 test rows
2. **Scale features** — divides `age` by 100 so both features sit roughly in the same `0–1` range as `affordability`
3. **Train a Keras baseline** — a single `Dense(1, activation='sigmoid')` layer (i.e. logistic regression / one perceptron with a sigmoid), compiled with `adam` + `binary_crossentropy`, trained for 5000 epochs → **100% test accuracy**, test loss ≈ `0.42`
4. **Extract the learned parameters** directly from the trained Keras layer (`model.get_weights()`) — the actual `w1`, `w2`, and `bias` the network converged to
5. **Reproduce the prediction by hand** — a manual `sigmoid()` + `prediction_function()` using just those extracted weights, confirming exactly what the Keras layer computes internally
6. **Implement gradient descent manually** — a `gradient_descent()` function that:
   - starts `w1 = w2 = 1`, `bias = 0`
   - computes predictions with a vectorized `sigmoid_numpy()`
   - computes log loss at each step
   - derives the gradients for `w1`, `w2`, and `bias` using `np.dot` across the whole batch
   - updates each parameter with a fixed learning rate (`0.5`)
   - loops until a max epoch count or a loss threshold is reached
7. **Compares the manually-trained weights** against the Keras-trained weights to see how close a fully from-scratch loop gets to the "real" model

## Concepts covered

- Logistic regression as a single neuron: weighted sum → sigmoid → probability
- Why feature scaling matters before training
- Reading a trained layer's actual weights and bias instead of treating it as a black box
- Deriving and coding the gradient descent update rule for logistic regression + log loss by hand, instead of just calling `.fit()`
- Vectorizing a full-batch weight update with `np.dot`, tying directly back into the [Matrix Basics](https://github.com/avrajyoti07/4DL_Matrix_Basics) project

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Keras `Dense(1, sigmoid)`, 5000 epochs | 100% | ~0.42 |

**Keras-learned parameters:** `w1 (age) ≈ 5.77`, `w2 (affordability) ≈ 1.20`, `bias ≈ -2.94`

## A note on a bug worth knowing about

The manual `log_loss()` function in this notebook has a small typo carried over from the [Loss & Cost Function](https://github.com/avrajyoti07/5DL_Loss_-_Cost_Function) project: the upper clipping step uses `max(i, 1 - epsilon)` where it should use `min(i, 1 - epsilon)`. As written, that line forces almost every predicted value up to `~0.999999999999999` regardless of what the model actually predicted — which is why the printed loss stays exactly constant across every epoch in the `gradient_descent()` output, instead of decreasing as training progresses. The weight updates themselves are still computed correctly (they don't depend on that buggy loss value), so the parameters still move — just know that the loss number printed during training isn't meaningful until that one `max` → `min` fix is made.

## Next steps

- [ ] Fix the `max`/`min` typo in `log_loss()` and re-run to see the loss actually decrease
- [ ] Let `gradient_descent()` run to full convergence and compare final `w1`, `w2`, `bias` against the Keras-trained values above
- [ ] Plot the loss curve over epochs
- [ ] Try different learning rates and see how convergence speed changes
- [ ] Extend to more features or a larger dataset

## Related projects in this series

- [1. Perceptron Demo](https://github.com/avrajyoti07/1DL_Perceptron_demo)
- [2. Hand Digits Classification](https://github.com/avrajyoti07/2DL_Hand_Digits_Classification)
- [3. Activation Functions](https://github.com/avrajyoti07/3DL_Activation_Function)
- [4. Matrix Basics](https://github.com/avrajyoti07/4DL_Matrix_Basics)
- [5. Loss & Cost Functions](https://github.com/avrajyoti07/5DL_Loss_-_Cost_Function)

## Tech stack

- Python 3
- `tensorflow` / `keras`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/avrajyoti07/6DL_Gradient_Discent.git
cd 6DL_Gradient_Discent
```

**2. Install dependencies**
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

**3. Run the notebook**
```bash
jupyter notebook 5_Gradient_Descent.ipynb
```
Run all cells top to bottom.

## Author

Built by [avrajyoti07](https://github.com/avrajyoti07) as part of a Deep Learning learning journey.
Feel free to fork, star, or open an issue if you spot something to improve!

## License

This project is open source under the [MIT License](LICENSE).
