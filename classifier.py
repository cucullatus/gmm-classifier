from generateData import * 
from gmm3 import *

X0train = generateDataCat('data/train.txt',0)
X1train = generateDataCat('data/train.txt',1)

# training
k0 = 4
k1 = 4
gmm0 = GMM(k0,X0train)
# train GMM0
gmm0.gmm(X0train,k0)

gmm1 = GMM(k1,X1train)
gmm1.gmm(X1train,k1)

X0dev = generateDataCat('data/dev.txt',0)
X1dev = generateDataCat('data/dev.txt',1)
Xdev = generateData('data/dev.txt')
ansDev = generateAns('data/dev.txt')

redev = generateResult(gmm0,gmm1,Xdev)
devacc = evaluate(redev,ansDev)
print 'Accuarcy on Dev dataset: ', devacc

# generate test result
Xtest = generateData('data/test.txt')
retest = generateResult(gmm0,gmm1,Xtest)
row, col = np.shape(Xtest)
print "row, col", row, " ",col
outf = open('result/result0.txt','w')

for i in range(row):
	line = ""
	for j in range(col):
		line += str(np.array(Xtest)[i][j])
		line += " "
	line += " "
	line += str(retest[i]+1)
	line += '\n'
	outf.write(line)	

outf.close()

