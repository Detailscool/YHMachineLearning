import matplotlib.pyplot as plt

def drawCurve(Reagentperday, CurrentLevel) :
    RegentLevel = []
    for i in range(15):
        RegentLevel.append(CurrentLevel)
        CurrentLevel = CurrentLevel * 0.5 + Reagentperday
    Time = [i for i in range(15)]
    print(Time, RegentLevel)
    plt.title('Reagentperday: %s CurrentLevel: %s ' % (Reagentperday, CurrentLevel))
    plt.plot(Time, RegentLevel)

if __name__ == '__main__':
    # for k in range(1, 10):
    for i in range(1, 4):
        drawCurve(1, i)
    plt.show()
