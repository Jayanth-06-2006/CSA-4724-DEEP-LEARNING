
"""Experiment 5: Gradient descent minimizing J(w) = (w-3)^2
Input: starting w=0, learning rate 0.4, 10 iterations
Output: iteration-wise w and cost values printed and plotted.
"""
import numpy as np
import matplotlib.pyplot as plt

def J(w): return (w-3)**2
def dJ(w): return 2*(w-3)

w = 0.0
lr = 0.4
iters = 10
ws, costs = [], []
for i in range(iters):
    ws.append(w)
    costs.append(J(w))
    grad = dJ(w)
    w = w - lr * grad
    print(f"Iter {i}: w={ws[-1]:.6f}, cost={costs[-1]:.6f}, grad={grad:.6f}")

print("Final w:", w, "Final cost:", J(w))
plt.plot(range(iters), costs, marker='o')
plt.xlabel("Iteration"); plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.savefig("exp5_gd_cost.png")
print("Saved exp5_gd_cost.png")
