import torch
import random
import matplotlib.pyplot as plt


w = torch.tensor(2., requires_grad=True)
b = torch.tensor(1., requires_grad=True)


def function(x: torch.tensor) -> torch.tensor:
    return w * x ** 2 + b


def backward_propagation(x: torch.tensor):
    x.backward()


def loss(x, y):
    return (x - y) ** 2


def gradient_descent(params: torch.tensor, learning_rate=0.01):
    with torch.no_grad():
        for param in params:
            param -= param.grad * learning_rate
            param.grad.zero_()


def answer(x: torch.tensor) -> torch.tensor:
    return 10000 * x ** 2 - 10000


for i in range(10000):
    variation = random.uniform(0, 1)
    X = function(variation)
    Y = answer(variation)
    l = loss(X, Y)
    backward_propagation(l)
    gradient_descent([w, b])
    if not (i + 1) % 100:
        print(w, b, l)

print(w, b)
