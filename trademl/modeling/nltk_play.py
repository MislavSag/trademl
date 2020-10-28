import pandas as pd
import yfinance as yf


# data
data = yf.download('SPY','2001-08-28','2020-10-25')

 
def trend_labeling(close, time, w):

    # init vars
    fp = close[0]  # represents the first price obtained by the algorithm
    xh = close[0]  # mark the highest price
    ht = time[0]  # mark the time when the highest price occurs
    xl = close[0]  #  mark the lowest price
    lt = time[0]  # mark the time when the lowest price occurs
    cid = 0  # mark the current direction of labeling
    fp_n = 0  # the index of the highest or lowest point obtained initially


    for i, (index, value) in enumerate(zip(time, close)):
        if value > (fp + (fp * w)):
            xh = value
            ht = index
            fp_n = i
            cid = 1
            print("defineprice " + str(fp), " highest_price   break  " + str(i), ", time: " +
                str(ht) + "   value: " + str(xh))
            break
        if value < (fp - (fp * w)):
            xl = value
            lt = index
            fp_n = i
            cid = -1
            print("defineprice " + str(fp), " highest_price   break  " + str(i), ", time: " +
                str(lt) + "   value: " + str(xl))
            break


    y = [ ]

    for i, (index, value) in enumerate(zip(time[(fp_n + 1):-1], close[(fp_n + 1):-1]), start=fp_n + 1):
        
        if cid > 0:
            if value > xh:
                xh = value
                ht = index
            if value < (xh - (xh * w)) and lt < ht:
                print(f'{index} { 1}')
                for date in time:
                    if date > lt and date <= ht:
                        y.append(1)
                xl = value
                lt = index
                cid = -1
        
        if cid < 0:
            if value < xl:
                xl = value
                lt = index
            if value > (xl + (xl * w)) and ht < lt:
                print(f'{index} { -1}')
                for date in time:
                    if date > ht and date <= lt:
                        y.append(-1)
                xh = value
                ht = index
                cid = 1

    # add last segment label that is opposite of last available
    y_last_segment = [y[-1] * -1] * (len(close) - len(y))
    y = y + y_last_segment    
    
    return y


y = trend_labeling(data['Close'].values.tolist(), data.index.tolist(), 0.15)
len(y)
