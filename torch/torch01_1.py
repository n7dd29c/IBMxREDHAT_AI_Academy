import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)  # 텐서로 변환
# print(x)                  # tensor([1., 2., 3.])
# print(x.shape)            # torch.Size([3])
# print(x.size())           # shape랑 똑같음

x = torch.FloatTensor(x).unsqueeze(1)   # unsqueeze : 차원을 늘려주는것, 숫자는 1이 들어가는 위치
# print(x)
# print(x.shape)            # torch.Size([3, 1])
# print(x.size())           # torch.Size([3, 1])

y = torch.FloatTensor(y).unsqueeze(1)
print(y.size())

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))  # 앞이 output, 뒤가 input
model = nn.Linear(1, 1)             # 위의 표현과 같음, 하지만 앞이 input, 뒤가 output
                                    # y = wx + b 가 아니었다 ㄷㄷ
                                    # y = xw + b 가 맞다, 데이터에 가중치를 곱하는거
                                    # 행렬연산은 곱하기의 순서를 바꾸면 값이 완전히 달라짐
                                    
#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.077)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()                   # default, 명시하지 않아도 적용 (훈련모드)
    optimizer.zero_grad()           # 기울기(gradient) 초기화
                                    # 각 배치마다 기울기를 초기화(0으로)해서,
                                    # 기울기 누적에 의한 문제 해결
    hypothesis = model(x_train)           # y = xw + b 를 정의, predict 와 같음
    loss = criterion(hypothesis, y_train) # loss = mse 와 같음
                                    # loss = mse() = ∑(y - hypothesis)^2/n
    loss.backward()                 # 기울기(gradient)값 까지만 계산
    optimizer.step()                # 가중치 갱신
    
    return loss.item()
    
epochs = 124
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}   loss : {}'.format(epoch, loss))
    
# 기울기 != 가중치

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x_test, y_test):
    model.eval()                    # dropout, BatchNomalization 을 쓰지 않겠다는 의미, 필수! (테스트모드)
    with torch.no_grad():           # 기울기(gradient) 갱신을 하지 않겠다, 고정
        y_predict = model(x)
        final_loss = criterion(y, y_predict)
    return final_loss.item()        # loss의 최종값

final_loss = evaluate(model, criterion, x, y)
results = model(torch.Tensor([[4]]))
print('최종 loss :', final_loss)
print('4의 예측값 :', results.item())

# 최종 loss : 9.754912025528029e-05
# 4의 예측값 : 3.9857630729675293