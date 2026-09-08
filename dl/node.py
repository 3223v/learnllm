class Node:
    def __init__(self,value, _backward = lambda:None):
        self.value = value
        self.grad = 0.0
        self._backward = _backward

    def backward(self):
        self._backward()

class Addop:
    @staticmethod
    def forward(a:Node, b:Node) ->Node:
        out = Node(a.value+b.value)

        def _backward():
            a.grad += out.grad
            b.grad += out.grad
        out._backward = _backward

        return out

class MatMulOp:
    @staticmethod
    def forward(w:Node,x:Node) ->Node:
        out = Node(w.value @ x.value)

        def _backward():

            w.grad += out.grad @ x.value.T

            x.grad += w.value.T @ out.grad

        out._backward = _backward
        return out
        

