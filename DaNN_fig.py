# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:14:43 2019

@author: 12709
"""

import matplotlib.pyplot as plt
#experiment 1
'''
file1 = open('C:/Users/12709/Documents/GitHub/DANN_P2R/test_l1_bs64_dr0.5_lin.txt','r')
file2 = open('C:/Users/12709/Documents/GitHub/DANN_P2R/train_l1_bs64_dr0.5_lin.txt','r')
epoch = []
training_acc = []
testing_acc = []
train_loss = []
test_loss = []
for line in file1:
    data = [s for s in line.split(' ')]
    epoch.append(float(data[0]))
    test_loss.append(float(data[1]))
    testing_acc.append(float(data[2]))
for line in file2:
    data = [s for s in line.split(' ')]
    training_acc.append(float(data[2]))
    train_loss.append(float(data[1]))

max1 = max(train_loss)
max2 = max(test_loss)
for i in range(len(train_loss)):
    train_loss[i] = train_loss[i]/max1
    test_loss[i] = test_loss[i]/max2

plt.gca().set_color_cycle(['red','black','blue','green','pink','magenta','yellow','cyan'])
plt.plot(epoch, training_acc)
plt.plot(epoch, testing_acc)
plt.plot(epoch, train_loss)
plt.plot(epoch, test_loss)
plt.legend(['training accuracy','testing accuracy', 'training loss', 'testing loss'], loc = 'right')
plt.xlabel('Epoch')
plt.savefig('DaNN_fig1.png')
plt.show()
'''
#experiment 2
'''
file1 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.5_lin.txt','r')
file2 = open('C:/Users/12709/Documents/GitHub/DANN_C2R/test_l1_bs64_dr0.5_lin.txt','r')
file3 = open('C:/Users/12709/Documents/GitHub/DANN_P2R/test_l1_bs64_dr0.5_lin.txt','r')
epoch = []
A2R_test_acc =[]
C2R_test_acc = []
P2R_test_acc = []
for line in file1:
     data = [s for s in line.split(' ')]
     epoch.append(float(data[0]))
     A2R_test_acc.append(float(data[2]))
for line in file2:
    data = [s for s in line.split(' ')]
    C2R_test_acc.append(float(data[2]))
for line in file3:
    data = [s for s in line.split(' ')]
    P2R_test_acc.append(float(data[2]))
plt.gca().set_color_cycle(['red','black','blue','green','pink','magenta','yellow','cyan'])
plt.plot(epoch, A2R_test_acc)
plt.plot(epoch, C2R_test_acc)
plt.plot(epoch, P2R_test_acc)
plt.legend(['A2R_test_acc','C2R_test_acc', 'P2R_test_acc'], loc = 'right')
plt.xlabel('Epoch')
plt.savefig('DaNN_fig2.png')
plt.show()
'''
#experiment 3
'''
file1 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l0.25_bs64_dr0.5_lin.txt','r')
file2 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.5_lin.txt','r')
file3 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l2_bs64_dr0.5_lin.txt','r')
file4 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l4_bs64_dr0.5_lin.txt','r')
epoch = []
acc1 = []
acc2 = []
acc3 = []
acc4 = []
for line in file1:
     data = [s for s in line.split(' ')]
     epoch.append(float(data[0]))
     acc1.append(float(data[2]))
for line in file2:
    data = [s for s in line.split(' ')]
    acc2.append(float(data[2]))
for line in file3:
    data = [s for s in line.split(' ')]
    acc3.append(float(data[2]))
for line in file4:
    data = [s for s in line.split(' ')]
    acc4.append(float(data[2]))
plt.gca().set_color_cycle(['red','black','blue','green','pink','magenta','yellow','cyan'])
plt.plot(epoch, acc1)
plt.plot(epoch, acc2)
plt.plot(epoch, acc3)
plt.plot(epoch, acc4)
plt.legend(['lambda = 0.25','lambda = 1', 'lambda = 2', 'lambda = 4'], loc = 'right')
plt.xlabel('Epoch')
plt.savefig('DaNN_fig3.png')
plt.show()
'''
#experiments 4

file1 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs32_dr0.5_lin.txt','r')
file2 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.5_lin.txt','r')
file3 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs128_dr0.5_lin.txt','r')
file4 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs256_dr0.5_lin.txt','r')
file5 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs512_dr0.5_lin.txt','r')
epoch = []
acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
for line in file1:
     data = [s for s in line.split(' ')]
     epoch.append(float(data[0]))
     acc1.append(float(data[2]))
for line in file2:
    data = [s for s in line.split(' ')]
    acc2.append(float(data[2]))
for line in file3:
    data = [s for s in line.split(' ')]
    acc3.append(float(data[2]))
cnt1 = 0
for line in file4:
    if cnt1<30:
        cnt1 +=1
        data = [s for s in line.split(' ')]
        acc4.append(float(data[2]))
cnt2 = 0
for line in file5:
    if cnt2<30:
        cnt2 +=1
        data = [s for s in line.split(' ')]
        acc5.append(float(data[2]))
plt.gca().set_color_cycle(['red','black','blue','green','pink','magenta','yellow','cyan'])
plt.plot(epoch, acc1)
plt.plot(epoch, acc2)
plt.plot(epoch, acc3)
plt.plot(epoch, acc4)
plt.plot(epoch, acc5)
plt.legend(['batch size = 32','batch size = 64', 'batch size = 128', 'batch size = 256','batch size = 512'], loc = 'right')
plt.xlabel('Epoch')
plt.savefig('DaNN_fig4.png')
plt.show()

#experiment5
'''
file1 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0_lin.txt','r')
file2 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.3_lin.txt','r')
file3 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.5_lin.txt','r')
file4 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.8_lin.txt','r')
file5 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.9_lin.txt','r')
file6 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr0.99_lin.txt','r')
file7 = open('C:/Users/12709/Documents/GitHub/DANN_A2R/test_l1_bs64_dr1_lin.txt','r')
epoch = []
acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
acc6 = []
acc7 = []
for line in file1:
     data = [s for s in line.split(' ')]
     epoch.append(float(data[0]))
     acc1.append(float(data[2]))
for line in file2:
    data = [s for s in line.split(' ')]
    acc2.append(float(data[2]))
for line in file3:
    data = [s for s in line.split(' ')]
    acc3.append(float(data[2]))
for line in file4:
    data = [s for s in line.split(' ')]
    acc4.append(float(data[2]))
for line in file5:
    data = [s for s in line.split(' ')]
    acc5.append(float(data[2]))
for line in file6:
    data = [s for s in line.split(' ')]
    acc6.append(float(data[2]))
for line in file7:
    data = [s for s in line.split(' ')]
    acc7.append(float(data[2]))
plt.gca().set_color_cycle(['red','black','blue','green','pink','magenta','yellow','cyan'])
plt.plot(epoch, acc1)
plt.plot(epoch, acc2)
plt.plot(epoch, acc3)
plt.plot(epoch, acc4)
plt.plot(epoch, acc5)
plt.plot(epoch, acc6)
plt.plot(epoch, acc7)
plt.legend(['drop rate = 0','drop rate = 0.3', 'drop rate = 0.5', 'drop rate = 0.8','drop rate = 0.9', 'drop rate = 0.99', 'drop rate = 1'], loc = 'right')
plt.xlabel('Epoch')
plt.savefig('DaNN_fig5.png')
plt.show()
'''