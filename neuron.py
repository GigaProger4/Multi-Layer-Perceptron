import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

class Neuron:
    def __init__(self, Data, RM_Data):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

        self.Data = Data
        self.RM_Data = RM_Data

        self.Learning_Rate = 0.04            #качество обучения(множитель изменения состояния ошибки и правилного ответа)
        self.epochs = 1000                  #количество эпох обучения

    def Feedforward(self, x1,x2,x3):
        h1 = sigmoid(x1 * self.w1 + x2 * self.w2 + x3 * self.w3 + self.b1)
        h2 = sigmoid(x1 * self.w4 + x2 * self.w5 + x3 * self.w6 + self.b2)
        h3 = sigmoid(x1 * self.w7 + x2 * self.w8 + x3 * self.w9 + self.b3)
        return(sigmoid(h1 * self.w10 + h2 * self.w11 + h3 * self.w12 + self.b4))

    def Neuro_Learning(self):
        for epoch in range (self.epochs + 1):
            for i in range (len(self.RM_Data)):

                h1 = self.Data[i][0] * self.w1 + self.Data[i][1] * self.w2 + self.Data[i][2] * self.w3 + self.b1
                h2 = self.Data[i][0] * self.w4 + self.Data[i][1] * self.w5 + self.Data[i][2] * self.w6 + self.b2
                h3 = self.Data[i][0] * self.w7 + self.Data[i][1] * self.w8 + self.Data[i][2] * self.w9 + self.b3

                o1 = (sigmoid(h1) * self.w10 + sigmoid(h2) * self.w11 + sigmoid(h3) * self.w12 + self.b4)
                py = sigmoid(o1)

                dL_dpy = -2 * (self.RM_Data[i] - py)

                #h1
                dpy_dh1 = self.w10 * deriv_sigmoid(o1)
                #h2
                dpy_dh2 = self.w11 * deriv_sigmoid(o1)
                #h3
                dpy_dh3 = self.w12 * deriv_sigmoid(o1)

                #h1
                dh1_dx1 = data[i][0] * deriv_sigmoid(h1)
                dh1_dx2 = data[i][1] * deriv_sigmoid(h1)
                dh1_dx3 = data[i][2] * deriv_sigmoid(h1)
                dh1_db1 = deriv_sigmoid(h1)

                #h2
                dh2_dx1 = data[i][0] * deriv_sigmoid(h2)
                dh2_dx2 = data[i][1] * deriv_sigmoid(h2)
                dh2_dx3 = data[i][2] * deriv_sigmoid(h2)
                dh2_db2 = deriv_sigmoid(h2)

                #h3
                dh3_dx1 = data[i][0] * deriv_sigmoid(h3)
                dh3_dx2 = data[i][1] * deriv_sigmoid(h3)
                dh3_dx3 = data[i][2] * deriv_sigmoid(h3)
                dh3_db3 = deriv_sigmoid(h3)

                #o1
                do1_dh1 = sigmoid(h1) * deriv_sigmoid(o1)
                do1_dh2 = sigmoid(h2) * deriv_sigmoid(o1)
                do1_dh3 = sigmoid(h3) * deriv_sigmoid(o1)
                do1_db4 = deriv_sigmoid(o1)

                # изменяем вес по пути w из x в h
                self.w1 -= dL_dpy * dpy_dh1 * dh1_dx1 * self.Learning_Rate
                self.w2 -= dL_dpy * dpy_dh1 * dh1_dx2 * self.Learning_Rate
                self.w3 -= dL_dpy * dpy_dh1 * dh1_dx3 * self.Learning_Rate
                self.b1 -= dL_dpy * dpy_dh1 * dh1_db1 * self.Learning_Rate

                self.w4 -= dL_dpy * dpy_dh2 * dh2_dx1 * self.Learning_Rate
                self.w5 -= dL_dpy * dpy_dh2 * dh2_dx2 * self.Learning_Rate
                self.w6 -= dL_dpy * dpy_dh2 * dh2_dx3 * self.Learning_Rate
                self.b2 -= dL_dpy * dpy_dh2 * dh2_db2 * self.Learning_Rate

                self.w7 -= dL_dpy * dpy_dh3 * dh3_dx1 * self.Learning_Rate
                self.w8 -= dL_dpy * dpy_dh3 * dh3_dx2 * self.Learning_Rate
                self.w9 -= dL_dpy * dpy_dh3 * dh3_dx3 * self.Learning_Rate
                self.b3 -= dL_dpy * dpy_dh3 * dh3_db3 * self.Learning_Rate

                # изменяем вес по пути w из h в o1
                self.w10 -= dL_dpy * do1_dh1 * self.Learning_Rate
                self.w11 -= dL_dpy * do1_dh2 * self.Learning_Rate
                self.w12 -= dL_dpy * do1_dh3 * self.Learning_Rate
                self.w10 -= dL_dpy * do1_db4 * self.Learning_Rate

            if (epoch % 10 == 0):
                sum = 0
                for i in range(len(self.RM_Data)):
                    sum += (self.RM_Data[i] - self.Feedforward(self.Data[i][0],self.Data[i][1],self.Data[i][2]))**2
                loss = sum/len(self.RM_Data)
                print("Epoch: %d loss: %.3f" % (epoch, loss))

# data = zp,rost,ves
data = [
    [4, -3, -4], #настя
    [24, 7, 5], #илья
    [-19, -4, -8], #катя
    [1, 1, 4]  # вася
]

real_data = [1,0,1,0]

neuron = Neuron( data,real_data)

# test0 = neuron.Feedforward(7,7,7)
neuron.Neuro_Learning()
# test1 = neuron.Feedforward(7,7,7)

mas = [0] * 3
mas[0] = input("введите зарплату человека : ")
mas[1] = input("введите рост человека : ")
mas[2] = input("введите вес человека : ")

result = neuron.Feedforward(int(mas[0]),int(mas[1]),int(mas[2]))

if(result > 0.5):
    print("я думаю это девушка с вероятностью : ", result * 100)
else:
    print("я думаю это парень с вероятностью : ", (1 - result) * 100)




