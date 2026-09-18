# x = 2
# target = 1

# weight = 0.32
# bias = 0.0

# learning_rate = 0.1

# prediction = (x * weight) + bias
# error = prediction - target

# new_weight = weight - (learning_rate * error * x)

# print("prediction:", prediction)
# print("error:", error)
# print("new weight:", new_weight)


# x = 2
# target = 1

# weight = 0.2
# bias = 0.0

# learning_rate = 0.1

# for epoch in range(10):

#     # Prediction
#     prediction = (x * weight) + bias

#     # Error
#     error = prediction - target

#     # Update weight
#     weight = weight - (learning_rate * error * x)

#     print(
#         f"Epoch {epoch + 1}: "
#         f"prediction={prediction:.4f}, "
#         f"error={error:.4f}, "
#         f"weight={weight:.4f}"
#     )

# data = [
#     (1, 2),
#     (2, 4),
#     (3, 6),
#     (4, 8),
# ]

# weight = 0.1
# learning_rate = 0.01

# for epoch in range(10):

#     for x, target in data:

#         prediction = x * weight
#         error = prediction - target

#         weight = weight - (learning_rate * error * x)

#     print(
#         f"Epoch {epoch + 1}: weight={weight:.4f}"
#     )

data = [
    ([1, 1], 5),
    ([2, 1], 7),
    ([1, 2], 8),
    ([2, 2], 10),
]

w1 = 0.1
w2 = 0.1

learning_rate = 0.01

for epoch in range(20):

    for inputs, target in data:

        x1, x2 = inputs

        prediction = (x1 * w1) + (x2 * w2)

        error = prediction - target

        w1 = w1 - (learning_rate * error * x1)
        w2 = w2 - (learning_rate * error * x2)

    print(
        f"Epoch {epoch + 1}: "
        f"w1={w1:.4f}, "
        f"w2={w2:.4f}"
    )