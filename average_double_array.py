collected_data_round = [[0, 1], [2, 3], [6, 1]]
collected_data_means = []

print(collected_data_round)

for i in range(0, len(collected_data_round) - 1):
    moy = 0
    # print(i)
    for j in collected_data_round:
        print(j[i])
        moy = moy + j[i]
    # print("total " + str(moy))
    moy = round(moy / len(collected_data_round))
    collected_data_means.append(moy)
    # print(moy)
    moy = 0
    # print("--")

print(collected_data_means)
