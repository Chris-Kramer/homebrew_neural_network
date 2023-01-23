import pandas as pd
from random import randint
cols = ["temp", "week", "rank", "price","buy"]

# Train
rows = []
for _ in range(250):
    select = randint(0, 4)
    no_rand_temp = [1, 2, 3, 4, 5]
    no_rand_weeks = [2, 3, 4, 5, 6]
    no_rand_price = [300, 400, 500, 1000, 1100] 
    # no buyers
    row_no = [no_rand_temp[select], no_rand_weeks[select],
              randint(10, 20), no_rand_price[select], 0]
    
    # Buyers
    yes_rand_temp = [15, 20, 25, 30, 35]
    yes_rand_weeks = [25, 30, 35, 40, 45]
    yes_rand_price = [25, 50, 75, 100, 150] 
    row_yes = [yes_rand_temp[select], yes_rand_weeks[select],
              randint(1, 10), yes_rand_price[select], 1]
    rows.append(row_no)
    rows.append(row_yes)
df = pd.DataFrame(columns= cols, data= rows)
df.to_csv("tickets_train.csv", index=False)

# Test
rows = []
for _ in range(100):
    select = randint(0, 4)
    no_rand_temp = [1, 2, 3, 4, 5]
    no_rand_weeks = [2, 3, 4, 5, 6]
    no_rand_price = [300, 400, 500, 1000, 1100] 
    # no buyers
    row_no = [no_rand_temp[select], no_rand_weeks[select],
              randint(10, 20), no_rand_price[select], 0]
    
    # Buyers
    yes_rand_temp = [15, 20, 25, 30, 35]
    yes_rand_weeks = [25, 30, 35, 40, 45]
    yes_rand_price = [25, 50, 75, 100, 150] 
    row_yes = [yes_rand_temp[select], yes_rand_weeks[select],
              randint(1, 10), yes_rand_price[select], 1]
    rows.append(row_no)
    rows.append(row_yes)
df = pd.DataFrame(columns= cols, data= rows)
df.to_csv("tickets_test.csv", index=False)