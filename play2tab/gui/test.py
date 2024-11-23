import numpy as np

L = lambda i: 2**(-i/12)

L_template = np.array([L(i) for i in range(23)])
# L_template = L_template - L_template[-1]
# print(L_template)

# print(1.4311/(2.7821-1.4311))
# print((L_template[0]-L_template[1])/(L_template[1]-L_template[2]))

i = 10
print((L_template[i+2] - L_template[i+1]) / (L_template[i+1] - L_template[i]))

c = 2**(-1/12)

x = 0.001

print((1-x*c**i)/(1-x*c**(i+2)))