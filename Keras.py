import os # lấy đường dẩn
import cv2 # xử lí ảnh và ma trận
import numpy as np # xử lí ma trận
from keras.layers import Dense, Flatten #dùng xử lý các layers của Keras
from keras.models import Sequential #dùng để tạo model dùng trong thuật toán Keras
from keras import metrics 
import keras # sử dụng thư viện Keras


# duong dan toi thu muc chua ảnh
PATH = 'C:\\date\\hoa\\hd\\'
PATH2 = 'C:\\date\\hoa\\hong\\'
#chuyen doi anh thanh ma tran
imgw = cv2.imread(PATH + '1.jpg')
# resize lai kich thuoc ma tran
img_resizew = cv2.resize(imgw, (1000, 1000))
# cho ma tran vao list tên listx ( khoi tao list)
listx = [img_resizew]
# vong for lay tung anh trong file bo vao list voi kieu ma tran
for i in range(2,21): # cho vòng lập chạy từ i = 2 đến i = 20
	img = cv2.imread(PATH + str(i) + '.jpg') # chuyển ảnh thành ma trận (ảnh lấy từ thư mục PATH và ảnh có tên i.jpg)
	img_resize = cv2.resize(img, (1000, 1000)) #resize lại kích thước của ma trận img thành 1000x1000
	listx.extend([img_resize]) # add ma trận img_resize vào list có tên listx
for i in range(1,21): # cho vòng lập chạy tờ i = 1 đên i = 20
	img = cv2.imread(PATH2 + str(i) + '.jpg')  # chuyển ảnh thành ma trận (ảnh lấy từ thư mục PATH2 và ảnh có tên i.jpg)
	img_resize = cv2.resize(img, (1000, 1000))	 #resize lại kích thước của ma trận img thành 1000x1000
	listx.extend([img_resize])  # add ma trận img_resize vào list có tên listx
# doi kieu list thanh kieu ma tran
x_train = np.asarray(listx)
x_train = x_train/255. # chia mỏi phần tử trong ma trận x_train cho 255
#in ra màng hình kích thước của ma trận x_train
print(x_train.shape)


"""
tao y train [1 0] hoa huong duong 
[0 1] khong phai hoa huong duong 
"""
listy = [[1,0]] # khởi tạo giá trị đầu cho list(chứa ma trận) có tên là listy
for i in range(1,40): # cho vòng lập chạy từ i = 1 đến i = 39
	if(i < 20):
		listy.extend([[1,0]]) # cho 20 phần tử ma trận đầu của list tên listy có giá [1,0]
	else:
		listy.extend([[0,1]]) # cho 20 phần tử ma trận tiếp theo của list tên listy có giá [0,1]
y_train =np.asarray(listy) # đổi kiểu list thành kiểu ma trận
print(y_train.shape) # in ra kích thước của ma trận tên y_train

""" da co tap du lieu tien hanh code"""
num_classes = 2 # 0 hoac 1
model = Sequential() # khởi tạo model có kiểu Sequential (tuyến tín)
#flatten: lam phang ma tran
model.add(Flatten(input_shape=(1000, 3000)))

#dense: tao ra layer, 8 So luong node trong hidden layer(cung la input layer),activation: ham kich hoat cua tung node
model.add(Dense(8, activation='relu'))

#ouput later co 2 node , softmax: tinh xac xuat cho 2 node 0 va 1 de dua ra kq cho anh  
model.add(Dense(num_classes, activation='softmax'))
#loss: ham mat mat categorical_crossentropy
#optimizer ham toi uu, sgd: gradient descent
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])
x_train.shape = (40,1000,3000)
# dua du lieu cho model hoc, 25: cho theo 2^n, epochs: so lan chay, moi lan chay epochs thi no chi chay qua moi diem du lieu 1 lan
model.fit(x_train, y_train, batch_size=25, epochs = 1)
# tinh do chinh xac cua model
score = model.evaluate(x_train, y_train)

print("Do chinh xac:", score[1]*100,"%")

# doc hinh thanh ma tran
img = cv2.imread(PATH + '1.jpg')# lay anh trong Path
# resize lai kich thuoc ma tran
img_resize = cv2.resize(imgw, (1000, 1000))
# cho ma tran vao list ( khoi tao list)
list = [img_resizew]# khoi tao gia tri dau tien cho list
# sau khi co model thi su dung model de test du lieu moi
for i in range(10,14):# lay 4 anh trong PATH
	img = cv2.imread(PATH + str(i) + '.jpg')
	img_resize = cv2.resize(img, (1000, 1000))# chinh lai kich thuoc cua anh
	list.extend([img_resize])# add matrix vao list
#tao ma tran
x_test = np.asarray(list)
#5: co 5 anh co kich thuoc 1000x3000
x_test.shape = (5,1000,3000)

print(model.predict(x_test)) # in ra giá trị nhản của tập x_test
			  





	