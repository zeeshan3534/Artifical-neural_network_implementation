
import numpy as np

from nnfs.datasets import spiral_data
X = inputs = [[1,2,3,2.5],[2,5,-1,2] ,[-1.5,2.7,3.3,-0.8] ]
X,y = spiral_data(100,3)
class Layer_Dense:
    def __init__(self,Inputs,neurons):
        self.weights = np.random.randn(Inputs,neurons)
        self.bias = np.zeros((1,neurons))
    def forward(self,inputs):
        output=np.dot(inputs,self.weights)+self.bias
        return output
# print(np.zeros((1,5), dtype = [('x','int'),('y','float')]))

#activation number 1
class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)



#Activation number 2

class Softmax_activation:
    def forward(self,inputs):
        self.exp= np.exp(inputs-(np.max(inputs, axis=1,keepdims=True)))
        self.sumofall = np.sum(inputs,axis=1,keepdims=True)
        self.out= self.exp/self.sumofall
        return self.out

class Loss:
    def Calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

def loss_CrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidence=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidence=np.sum(y_pred_clipped*y_pred,axis=1)




L1=Layer_Dense(2,5)
act=Activation_ReLU()
out=L1.forward(X)
act.forward(out)
# print(act.output)
softmax= Softmax_activation()
print(softmax.forward(out))