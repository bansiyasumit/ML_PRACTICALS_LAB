import matplotlib.pyplot as plt


temperatures = [30, 32, 31, 29, 28, 27, 26, 25, 26, 27]


def first_order_moving_average(data):
    return [round((data[i] + data[i-1]) / 2, 2) for i in range(1, len(data))]


def second_order_moving_average(data):
    return [round((data[i] + data[i-1] + data[i-2]) / 3, 2) for i in range(2, len(data))]


filtered_1st = first_order_moving_average(temperatures)
filtered_2nd = second_order_moving_average(temperatures)


print("📊 Original Temperatures:")
print(temperatures)
print("\n✅ 1st-Order Moving Average:")
print(filtered_1st)
print("\n✅ 2nd-Order Moving Average:")
print(filtered_2nd)


days = list(range(1, len(temperatures)+1))

plt.figure(figsize=(10, 6))
plt.plot(days, temperatures, label='Original', marker='o')
plt.plot(days[1:], filtered_1st, label='1st-order MA', marker='s')
plt.plot(days[2:], filtered_2nd, label='2nd-order MA', marker='^')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Smoothing with Moving Average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
