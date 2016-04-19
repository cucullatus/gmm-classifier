#!/usr/bin/python
#coding:utf-8
"""
  GMM classification
  one GMM per class, tuned by maximum likelihood
  至于如何来调每个 GMM中components的个数 这个就如同 k-means如何选择合适的k的大小一样
"""
import numpy as np

X = np.mat([[1,2,3,4],[4,3,2,1],[5,6,7,8],[8,7,6,5],[4,5,6,7]])

class GMM(object):

	"""docstring for GMM algorithm"""
	def __init__(self, K_or_centroids, X):
		super(GMM, self).__init__()
		self.N, self.D = np.shape(X)
		self.Lprev = -np.inf
		self.L = 0.0
		self.threshold = 1e-7
		self.alpha = 1e-3
		if np.isscalar(K_or_centroids):
                        self.K = K_or_centroids
                        #randomly pick centroids
                        rndp = np.random.permutation(self.N) # 选 top-K 个 points 做 centroids
#			rndp = np.array([2,4,0,1,3])
                        self.centroids = X[rndp[0:self.K]] # 前 K 行
                else:
                        self.K = np.shape(K_or_centroids)[0]
                        self.centroids = K_or_centroids

		self.pMiu = self.centroids
                self.pPi = np.zeros((1,self.K))
                self.pSigma = np.zeros((self.K, self.D, self.D))

	def gmm(self,X, K_or_centroids):# Here, X is for training
		"""
		X: N by D matrix
		K_or_centroids:
		Px: N x K matrix indicating the probability of each component generating each point
		MODEL: a structure containing the parameters for a GMM:
			MODEL.Miu: K x D matrix
			MODEL.Sigma: D x D x K matrix
			MODEL.Pi: 1 x K vector
		"""

		# initial values
		[self.pMiu, self.pPi, self.pSigma] = self.init_params(X)
		it = 0
		while True:
			print "it: ", it
			it+=1
			Px = self.calc_prob(X) # 计算后验概率 N by K

			# new value for pGamma (N by K)
			# pGamma[i,k] = \frac{Px[i,k]}{sum_{j=1}^K(Px(i,j))}
			# i = 0,1,...,N-1  k = 0,1,...,K-1
			# pGamma[i,k] 对Px[i,k] 进行归一化 存的是 point i 由 component k 生成的概率 或者说把 point i的多少比例(比如 0.3个point i)分到 component k 中去
			pGamma = np.multiply(Px , np.kron(np.ones((self.N,1)),self.pPi)) # instead of repmat in matlab
			# pGamma N by K
			pGamma = pGamma / np.kron(np.ones((1,self.K)),np.transpose(np.mat(np.sum(pGamma,1))))

			# new value for parameters of each component
			Nk = np.sum(pGamma,0)
			print "Nk[0]: ", np.array(Nk)[0]
			Nk = np.array(Nk)[0]
			self.pMiu = np.diag(1./Nk)* np.transpose(pGamma) * X

			print "self.pMiu: ", self.pMiu

			self.pPi = Nk / self.N

			print "self.pPi: ", self.pPi

			for kk in range(self.K):
				Xshift = X - np.kron(np.ones((self.N,1)), self.pMiu[kk,:])
				print "Xshift: ", Xshift, np.shape(Xshift)
				candidateSigma=(np.mat(np.transpose(Xshift)) * np.mat(np.diag(np.array(pGamma[:,kk])[:,0])) * np.mat(Xshift)) / Nk[kk]
				print "np.diag(np.array(pGamma[:,kk])[:,0]: ", np.diag(np.array(pGamma[:,kk])[:,0])
				print "Nk[kk]: ", Nk[kk], np.shape(Nk[kk])

				print "candidateSigma: ", candidateSigma
				self.pSigma[kk,:,:] = self.regular(candidateSigma)
				print "regular candidateSigma: ", self.regular(candidateSigma)

			print "self.pSigma: ", self.pSigma

			# check for convergence
			self.L = np.sum(np.log(Px*np.transpose(np.mat(self.pPi))))
			print "self.L, self.Lprev: ", self.L, self.Lprev

			if (self.L - self.Lprev < self.threshold) or it >= 200:
				print "self.L, self.Lprev: ", self.L, self.Lprev
				break
			self.Lprev = self.L

	def init_params(self,X): # X is trainX
		pMiu = self.centroids
		pPi = np.zeros((1,self.K))
		pSigma = np.zeros((self.K, self.D, self.D))
		# hard assignment x to each centroids
		distmat = np.kron(np.ones((1,self.K)), np.sum(np.multiply(X,X),1)) + np.kron(np.ones((self.N,1)),np.transpose(np.sum(np.multiply(pMiu,pMiu),1))) - 2*X*np.transpose(pMiu)
		labels = np.array(np.transpose(np.argmin(distmat,1)))[0]# transpose(N by 1) -> 1 by N

		print '---init---'
		for k in range(self.K):
			Xk = X[labels==k]
			print Xk
			pPi[0,k] = np.shape(Xk)[0] / (self.N+0.0)
			covm = np.cov(Xk,rowvar=0)

			pSigma[k,:,:] = self.regular(np.cov(Xk,rowvar=0))
			print "k = ",k, "pSigma[k,:,:]: ", pSigma[k,:,:]
		return (pMiu, pPi, pSigma)

	def regular(self,sigma):
		if len(np.shape(sigma))==0:
                	sigma = np.array([sigma])
		else :
			sigma = np.array(sigma)
		sigma[np.diag(np.diag(sigma)<self.alpha)] = 0.0

		return  sigma + self.alpha * np.identity(self.D)


	def calc_prob(self,X):
		# 返回Px 1 by N
		# N(x_i|\mu_k,\Sigma_k) = \frac{1}{(2*pi)^{\frac{d}{2}} |\Sigma|^\frac{1}{2}}
				# exp\{ -\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\}
		Px = np.zeros((self.N,self.K))

		for k in range(self.K):
			# 计算 N(X|\mu_k,\Sigma_k)
			Xshift = X - np.kron(np.ones((self.N,1)),self.pMiu[k]) # Xshift N by D
			inv_pSigma = np.linalg.inv(self.pSigma[k,:,:]) # D by D
			tmp = (Xshift * inv_pSigma * np.transpose(Xshift)).diagonal()
			tmp = tmp[0] # 1 by N
			coef = (2*np.pi)**(-self.D/2)*np.sqrt(np.linalg.det(inv_pSigma))
			Px[:,k] = coef*np.exp(-0.5*np.array(tmp))
		print "calc_prob Px: ", Px
		return Px
		# Px[i,k] = N(x_i|\mu_k, \Sigma_k)
		# pGamma[i,k] 对Px[i,k] 进行归一化 存的是 point i 由 component k 生成的概率 或者说把 point i的多少比例(比如 0.3个point i)分到 component k 中去


def calc_prob(gmmModel,testX):
        # 返回Px 1 by N
        # N(x_i|\mu_k,\Sigma_k) = \frac{1}{(2*pi)^{\frac{d}{2}} |\Sigma|^\frac{1}{2}}
                        # exp\{ -\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\}
        N = np.shape(testX)[0]
        Px = np.zeros((N,gmmModel.K))

        for k in range(gmmModel.K):
                # 计算 N(X|\mu_k,\Sigma_k)
                Xshift = testX - np.kron(np.ones((N,1)),gmmModel.pMiu[k]) # Xshift N by D
                inv_pSigma = np.linalg.inv(gmmModel.pSigma[k,:,:]) # D by D
                tmp = (Xshift * inv_pSigma * np.transpose(Xshift)).diagonal()
                tmp = tmp[0] # 1 by N
                coef = (2*np.pi)**(-gmmModel.D/2)*np.sqrt(np.linalg.det(inv_pSigma))
                Px[:,k] = coef*np.exp(-0.5*np.array(tmp))
        print "calc_prob Px: ", Px
        return Px
        # Px[i,k] = N(x_i|\mu_k, \Sigma_k)
        # pGamma[i,k] 对Px[i,k] 进行归一化 存的是 point i 由 component k 生成的概率 或者说把 point i>的多少比例(比如 0.3个point i)分到 component k 中去


def predictIndx(trainedGMMinstance,newX):
	"""
	Px: N by K 取第i行最大的值的 index 作为 point i 的 component
	"""
	Px = calc_prob(trainedGMMinstance,newX)
	print "np.argmax(Px,1): ", np.argmax(Px,1)
	return np.argmax(Px,1)

def predictValue(trainedGMMinstance, newX):
	Px = calc_prob(trainedGMMinstance,newX)
	print "np.max(Px,1): ", np.max(Px,1)
        return np.max(Px,1)

def evaluate(re,ans):# re, ans are two list
    correct = 0
    for i in range(len(re)):
            if re[i] + 1 == ans[i]:
                    correct += 1
    acc = (0.0+correct) / len(re)
    return acc

if __name__=="__main__":
    # prepare data
    X = np.mat([[1,2,3,4],[4,3,2,1],[5,6,7,8],[8,7,6,5],[4,5,6,7]])
    #N,D = np.shape(X)
    gmm = GMM(3,X)
    gmm.gmm(X,3)
    testX = np.mat([[1,2,3,4],[4,3,2,1],[5,6,7,8],[8,7,6,5]])
    predictIndx(gmm,testX)
    predictValue(gmm, testX)
