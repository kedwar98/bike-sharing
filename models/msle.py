#Preliminary error function used for reported milestone results

def msle(y, ypred):

    act = np.array([np.log(i + 1) for i in y])
    pred = np.array([np.log(i + 1) for i in ypred])
    sq = (act - pred) ** 2
    return np.mean(sq) #take mean; divide by number of passed in points