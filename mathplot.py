import matplotlib.pyplot as plt
import numpy as np

x=np.array([2020,2021,2022,2023,2024,2025])
y = np.array([7,8,9,10,14,20])
y_fahrenheit = (y * 9/5) + 32
k = x + 273.15


plt.title('Temperature in helsinki in last 6 years')


# First subplot (Celsius)
plt.subplot(1, 3, 1)
plt.plot(x,y, 'bo--', label = 'celsius')
plt.xlabel('Years')
plt.ylabel('Temperature (°C)')
plt.legend()

# Second subplot (Fahrenheit)
plt.subplot(1, 3, 2)
plt.plot(x,y_fahrenheit, 'r*-',label = 'fahrenheit')
plt.xlabel('Years')
plt.ylabel('Temperature (°F)')
#it gives lable to lines
plt.legend()

plt.subplots_adjust(wspace=0.4)  # Increase horizontal space between subplots

# Third subplot (Kelvin)
plt.subplot(1, 3, 3)
plt.plot(x,k, 'bo--', label = 'kelwin')
plt.xlabel('Years')
plt.ylabel('Temperature (°K)')
plt.legend()

plt.savefig('pic.png')#to save graph pic

#printing of graph
plt.tight_layout()
plt.show()