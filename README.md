# Compress Sensing

Based on the Equation
### y = CΨs
where
y = measurement,
C = constant value assigned to random samples (sample size * # of data matrix),
s = sparse,

I have generated simple frequency to understand compess sensing.
Basic idea for compress sensing is that as long as we have certain amount of data from any signals(images, frequencies, etc.), we can reconstruct those data to restore original signals status using Fourier Transform. ![image.png](attachment:image.png)
(image from professor Steve Brunton Youtube Vedio:https://www.youtube.com/watch?v=SbU1pahbbkc&ab_channel=SteveBrunton)


```python
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from scipy import fftpack as fft
from sklearn.linear_model import Lasso
from PIL import Image, ImageOps
```


```python
# Initializing variables & signals needed for Compressed Sensing
n = 2000 # Number of data
ss = np.floor(n * 0.1).astype(int) # Limiting sample size to be 10% of data
t = np.linspace(0, 1/16, n) # Time frame
# 2 simple frequencies
y = np.cos(108 * 2 * np.pi * t) + 2 * np.sin(398 * 2 * np.pi * t)
# for more complex frequency
#y = np.cos(2 * 108 * np.pi * t) + 2 * np.sin(2 * 198 * np.pi * t) \
#    + np.sin(2 * 263 * np.pi * t) + np.cos(2 * 77 * np.pi * t) \
#    + np.cos(2 * 300 * np.pi * t)
ranIndex = np.random.randint(0, n, ss) # n * 0.1 randomly choose time intervals
sY = y[ranIndex] # sample y values corresponds to random indices
```


```python
# Shows how original signals looks like and locations of random samples
# Also shows 200 randomly chosen samples
plt.figure(figsize = (10, 4))
plt.plot(t, y)
plt.plot(t[ranIndex], sY, 'r.')
plt.title("Original Frequency")
plt.show()
# Shows how original signal transform would be like
plt.figure(figsize = (10, 4))
plt.plot(t, fft.dct(y, norm = 'ortho')) # plot cosine transform with ortho norm
plt.title("DCT Transform of Original Signal")
plt.show()
```



![png](output_3_0.png)





![png](output_3_1.png)




```python
# Create matrix C and apply discrete cosine transform to get Theta
c = np.eye(n)[ranIndex, :] # Creates s * n matrix C
print(c.shape)
print(ranIndex.shape)
theta = fft.dct(c) # Theta = C matrix * fft
```

    (200, 2000)
    (200,)


## Use L_1 Norm
By using L_1 Norm, we can minimize our transformed data to be mostly 0, which makes them to be sparse. We are not using L_2 norm, which is commonly used in other applications, because we want to make most of transformed data to be 0, and only keep essential data needed for restoration.
![image.png](attachment:image.png)
Image copied from Quora: https://www.quora.com/When-would-you-chose-L1-norm-over-L2-norm


```python
# Use Lasso for L_1 Minimization to get sparsity s
#print(theta.shape, sY.shape)
mini = Lasso(alpha = 0.001)
mini.fit(theta, sY)
s = mini.coef_ # Sparse s
#s.shape
```


```python
# Print transformed sample into graph to comapre with original
plt.figure(figsize = (10, 4))
plt.plot(t, s)
plt.title("transformed Sample")
plt.show()
```



![png](output_7_0.png)




```python
# Reverse the DCT to reconstruct signal into frequency
reform = fft.idct(s)
```


```python
plt.figure(figsize = (10, 4))
plt.plot(t, y)
plt.plot(t[ranIndex], sY, 'r.')
plt.title("Original Frequency")
plt.show()

plt.figure(figsize = (10, 4))
plt.plot(t, reform)
plt.title("Reformed Frequency")
plt.show()
```



![png](output_9_0.png)





![png](output_9_1.png)




```python
# Testing for various alpha penalty
alpha = [0.1, 0.01, 0.001, 0.0001, 0.3, 0.03, 0.003]
plt.figure(figsize = (10, 4))
plt.plot(t, y)
plt.plot(t[ranIndex], sY, 'r.')
plt.title("Original Frequency")
plt.show()
#coef = []
reformList = []
for a in alpha:
    mini = Lasso(alpha = a)
    mini.fit(theta, sY)
    s = mini.coef_
    #coef.append(s)
    reform = fft.idct(s)

    plt.figure(figsize = (10, 4))
    plt.plot(t, reform)
    plt.title("Reformed Frequency (alpha = %.4f)" %a)
    plt.show()
```



![png](output_10_0.png)





![png](output_10_1.png)





![png](output_10_2.png)





![png](output_10_3.png)



    C:\Users\Owner\anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.685e-02, tolerance: 4.466e-02
      model = cd_fast.enet_coordinate_descent(




![png](output_10_5.png)





![png](output_10_6.png)





![png](output_10_7.png)





![png](output_10_8.png)




```python
plt.plot(s)
```




    [<matplotlib.lines.Line2D at 0x1e27ce9f250>]





![png](output_11_1.png)




```python

```

# Image Compress Sensing

After Working on frequencies, I wanted to know how would it work out in 2D Array, as frequencies were 1D.

Basic approach should be similar to frequency Compress Sensing.
Like frequency, I will use 20 % to be my sample size


```python
# Importing images from local machine
image = Image.open("image1.jpg")
image = ImageOps.grayscale(image) # Make 3D rgb imgage into GrayScale 2D array
imgArr = np.asarray(image) # Represent image to be array
imgArr.shape
```




    (164, 307)




```python
#Show how image looks lie
plt.imshow(imgArr)
plt.title("Original Img")
plt.show()
plt.plot(imgArr)
plt.title("Plotting img array")
plt.show()
```



![png](output_15_0.png)





![png](output_15_1.png)




```python
# Descrete cosine transform of original image
plt.imshow(fft.dctn(imgArr, norm = 'ortho'))
plt.title("DCT of Original Imgae")
plt.show()
plt.plot(fft.dctn(imgArr, norm = 'ortho'))
plt.title("DCT Plot of Original Imgae")
plt.show()
```



![png](output_16_0.png)





![png](output_16_1.png)



## Set Variables

#### After using dctn, added normal 1d dct to see differences


```python
## For dctn/dct
n, m = imgArr.shape   #Setting n & m to be width and height of the arrag
sampleSz = np.floor(n * m * 0.2).astype(int)   #
ranIndex = np.random.randint(0, n * m, sampleSz)
print(ranIndex.shape)
sampleImgArr = imgArr.flatten()[ranIndex]
sampleImgArr2 = imgArr.flatten()[ranIndex]

# Both reshape and np.expand_dims works
sampleImgArr = np.reshape(sampleImgArr, (sampleSz, 1))   # Now arr.shape = (sampleSz, 1)
sampleImgArr2 = np.reshape(sampleImgArr, (sampleSz, 1))
```

    (10069,)


## Fetch sample datas into dct


```python
## Create C constant matrix
C = np.eye(n*m)[ranIndex, :] #creates shape = (10069, 50348), which is (sampleSz, total sample)
C3D = np.reshape(C, (sampleSz, n, m))
theta = fft.dctn(C3D, norm = 'ortho', axes = [1,2])

## Create C2 constant matrix
C2 = np.eye(n*m)[ranIndex2, :] #creates shape = (10069, 50348), which is (sampleSz, total sample)
theta2 = fft.dct(C2, norm = 'ortho')

print(C.shape, theta.shape)
print(C2.shape, theta2.shape)

```

    (10069, 50348) (10069, 164, 307)
    (10069, 50348) (10069, 50348)


## Use L1 Normalization to get sparse vector


```python
# Use Lasso for L_1 Minimization to get sparsity s for dctn
sampleImgArr = sampleImgArr.squeeze()
theta = np.reshape(theta, (sampleSz, (n*m)))
mini = Lasso(alpha = 0.0001)
mini.fit(theta, sampleImgArr)
s = mini.coef_ # Sparse s


```


```python
# Minimization for dct
mini2 = Lasso(alpha = 0.0001)
mini2.fit(theta2, sampleImgArr2)
s2 = mini2.coef_ # Sparse s
```

    C:\Users\Owner\anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:647: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.793e+03, tolerance: 2.792e+03
      model = cd_fast.enet_coordinate_descent(


## Reform the Image


```python
reform = fft.idctn(s.reshape(n, m))
reform2 = fft.idct(s2)
```


```python
plt.imshow(imgArr)
plt.title("Original Image")

reform = np.reshape(reform, (n, m))
plt.imshow(reform)
plt.title("Reformed Img Using dctn")
plt.show()

#reform2 = np.reshape(reform2, (n, m))
#plt.imshow(reform2)
#plt.title("Reformed Img Using dct")
#plt.show()
```



![png](output_26_0.png)




```python
plt.plot(s)
plt.title("Sparsity s for dctn")
plt.show()
#plt.plot(2)
#plt.title("Sparsity s for dct")
#plt.show()
```



![png](output_27_0.png)



## Check for Std. Error (Frobenius Norm)

Wanted to check how far are we off using only 20 % of data.

Also check how much dctn is better than dct. However, result turned out they are the same


```python
stdErr_dctn = np.linalg.norm(reform - imgArr, 'fro')
print(stdErr_dctn)
#stdErr_dct = np.linalg.norm(reform2 - imgArr, 'fro')

#print(stdErr_dctn == stdErr_dct)
```

    3990835.8490865617



```python

```


```python
# Testing for various alpha penalty
alpha = [0.001, 0.0001, 0.0003]
reformList = []
for a in alpha:
    mini = Lasso(alpha = a)
    mini.fit(theta, sampleImgArr)
    s = mini.coef_
    #coef.append(s)
    reform = fft.idctn(s.reshape(n, m))
    reform = np.reshape(reform, (n, m))
    plt.figure(figsize = (10, 4))
    plt.imshow(reform)
    plt.title("Reformed Frequency (alpha = %.4f)" %a)
    plt.show()
```



![png](output_31_0.png)





![png](output_31_1.png)





![png](output_31_2.png)




```python

```
