import numpy as np
import onnx
from onnx_tf.backend import prepare
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load the model and sample inputs and outputs
X_test = np.zeros((1001, 20))
xvals = np.linspace(-1, 1, 1001)
X_test[:, 0] = xvals
model = onnx.load('../best_saved_model/search_best.onnx')
JMP = 0.8
t1 = -norm.ppf(0.9) + JMP * (xvals > 0)
t2 = JMP * (xvals > 0)
t3 = norm.ppf(0.9) + JMP * (xvals > 0)
truth = [t1, t2, t3]
qs = [0.1, 0.5, 0.9]
color = ['green', 'red', 'yellow']
# Run the model with an onnx backend and verify the results
yt = []
for i in range(X_test.shape[0]):
    a = np.array(prepare(model).run(np.expand_dims(X_test[i], 0)))
    yt.append(a[0][0])
yt = np.array(yt)
for idx, (q, t, c) in enumerate(list(zip(qs, truth, color))):
    plt.plot(xvals, yt[:, idx], label=q, color=c)
    plt.plot(xvals, t, label=q, color=c)
plt.legend()
plt.savefig('../best_saved_model/search_best.png')
plt.show()
