import numpy as np
import pickle
from src.activation import *

class customize:
    def __init__(self):
        print('\n특정 네트워크를 만들고 저장하기 위한 코드입니다')
        print('우선 fully connected network 만 고려합니다\n')

        print('네트워크의 파라미터를 지정해주세요')
        self.input_size = int(input('입력데이터의 크기를 입력하세요: '))
        self.hidden_num = int(input('hidden layer 의 개수를 입력하세요(모두 같은 뉴런수를 가진다 가정): '))
        self.hidden_size = int(input('hidden layer 안의 뉴런수를 입력하세요: '))
        self.output_size = int(input('결과의 클래스 수를 입력하세요: '))

        self.weights = []
        print('가중치를 지정해주세요(단순 random 초기화만 고려합니다)')
        for i in range(self.hidden_num):
            if i == 0: # 처음에는 input_size 에 지배를 받으니까
                W = np.random.randn(self.input_size, self.hidden_size)
                b = np.random.randn(self.hidden_size)
            elif i == self.hidden_num-1: # 위와 마찬가지
                W = np.random.randn(self.hidden_size, self.output_size)
                b = np.random.randn(self.output_size)
            else : # 보통의 때
                W = np.random.randn(self.hidden_size, self.hidden_size)
                b = np.random.randn(self.hidden_size)

            self.weights.append(W)
            self.weights.append(b)

        self.layers = []
        W, b = self.weights[:2]
        j = 0
        print('각 층의 layer를 정해주세요')
        for i in range(self.hidden_num):
            cmd = input(str(i+1) + '번째 층의 layer 를 입력하세요: ')

            if cmd == 'Affine':
                self.layers.append(Affine(W, b))
                j += 1
                W, b = self.weights[j: j+2]

            elif cmd == 'Sigmoid':
                self.layers.append(Sigmoid())
                j += 1
                W, b = self.weights[j: j+2]


            for layer in self.layers:
                print(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x
        

        






