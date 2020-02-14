import matplotlib.pyplot as plt
import numpy as np

# x = np.array([0,1,2,3,4])
#
# y1 = np.array([38616,  9609, 12777,    67,     0])#/61069
# y2 = np.array([52913,   807,  1233,  4661,  1455])#/61069
# y3 = np.array([56855,   228,   280,   646,  3060])#/61069
#
# y1 = np.array([    1,  9609, 12777,    67,     0])#/22454
# y2 = np.array([14298,   807,  1233,  4661,  1455])#/22454
# y3 = np.array([18240,   228,   280,   646,  3060])#/22454
#
#
# my_xticks = ['0','1-9','10-29','30-99', '100+']
# plt.xticks(x, my_xticks)
# plt.plot(x, y1, linewidth=0.2, markersize=6, marker='o')
# plt.plot(x, y2, linewidth=0.2, markersize=6, marker='o')
# plt.plot(x, y3, linewidth=0.2, markersize=6, marker='o')
#
#
# plt.legend(['Pixel tracks', 'ECal hits', 'HCal hits'])
# plt.xlabel('Number of xyz')
# plt.ylabel('Frequency')
# plt.title('Distribution of data')
#
# plt.show()



x1 = np.array([1,2,3,4,5,6,7,8,9,11,14,17.5,21.5,36.5,50])
x2 = np.array([   1,    2,    3,    4,    5,    6,    7,    8,    9,   (10+12.)/2,
                  (13.+15)/2,   (16.+19)/2,   (20.+24)/2,   (25.+49)/2, 50])


#[0,10,20,50,100,200,300,400,500,600,700,800,900,10000]
x3 = np.array([(0.+9)/2,(10+19.)/2,(20.+49)/2,(50.+99)/2,(100.+199)/2,(200.+299)/2,(300.+399)/2,(400.+499)/2,(500.+599)/2,(600.+699)/2,(700.+799)/2,(800.+899)/2,1000])

# y1 = np.array([38616,  9609, 12777,    67,     0])#/61069
# y2 = np.array([52913,   807,  1233,  4661,  1455])#/61069
# y3 = np.array([56855,   228,   280,   646,  3060])#/61069

y1 = np.array([1607, 1086,  891,  809,  753,  906, 1005, 1263, 1289, 4335, 3916,
       2970, 1305,  318,    0])/22453


y2 = np.array([ 807,  631, 2392, 2871,  949,  383,   97,   21,    5,    0,    0,
          0,    0])/8156


y3 = np.array([228, 162, 347, 417, 691, 523, 384, 466, 440, 352, 185,  17,   2])/4214






# my_xticks = ['0','1-9','10-29','30-99', '100+']


my_xticks1 = ['1','2','3','4','5','6','7','8','9','10-12','13-15','16-19','19-24','24-49','50+']
my_xticks1 = ['1','','3','','5','','7','','9','10-12','13-15','16-19','19-24','24-49','50+']


my_xticks2 = [   '1',   '2',    '3',    '4',    '5',    '6',    '7',    '8',    '9',   '10-12',
                  '13-15',   '16-19',   '20-24',   '25-49', '50+']
my_xticks2 = [   '1',   '',    '3',    '',    '5',    '',    '7',    '',    '',   '10-12',
                  '13-15',   '16-19',   '20-24',   '25-49', '50+']

my_xticks3 = [   '1',    '2',    '3',    '4',    '5',    '6',    '7',    '8',    '9',   '10-14',   '15-19',
         '20-49',   '50-100',  '100+']
my_xticks3 = [   '1',    '',    '3',    '',    '5',    '',    '7',    '',    '',   '10-14',   '15-19',
         '20-49',   '50-100',  '100+']


# my_xticks3 = ['0-9','10-19','20-49','50-99','100-199','200-299','300-399','400-499','500-599','600-699','700-799','800-899','900-999','1000+']



# print(len(y1), len(my_xticks), len(x))
# 0/0
# plt.xticks(x1, my_xticks)
# plt.xticks(x2, my_xticks2)
# plt.xticks(x3, my_xticks3)
# plt.plot(x1, y1, linewidth=0.2, markersize=6, marker='o', c='#1f77b4')
plt.plot(x3, y2, linewidth=0.2, markersize=6, marker='o', c='#ff7f0e')
# plt.plot(x3, y3, linewidth=0.2, markersize=6, marker='o', c='#2ca02c')
# plt.plot(x, y2, linewidth=0.2, markersize=6, marker='o')
# plt.plot(x, y3, linewidth=0.2, markersize=6, marker='o')


# plt.legend(['Pixel tracks'])
# plt.xlabel('Number of pixel tracks')
plt.xlabel('Number of ecal hits')
# plt.xlabel('Number of hcal hits')
plt.ylabel('Frequency')
plt.title('Distribution of data')

plt.show()