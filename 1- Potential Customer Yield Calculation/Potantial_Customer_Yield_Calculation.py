## Potential Customer Yield Calculation with Rule-Based Classification

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

df = pd.read_csv("Datasets/persona.csv")
df.head()

#Examining data

df["SOURCE"].nunique()
df['SOURCE'].value_counts().plot(kind='bar')
plt.show()

df["PRICE"].nunique()
df["PRICE"].value_counts()

## Country-Based Sales
df["COUNTRY"].value_counts()

#Total Earnings by Country

df.groupby("COUNTRY").agg({"PRICE":"sum"})

#Platform-Based Sales

df["SOURCE"].value_counts()

#Price Averages by country

df.groupby("COUNTRY").agg({"PRICE":"mean"})

#Price Averages by platform

df.groupby("SOURCE").agg({"PRICE":"mean"})

#Price averages - COUNTRY-SOURCE breakdown

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})

#Average earnings in COUNTRY, SOURCE, SEX, AGE breakdown
df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})

#Sorting by Price

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE","customers_level_based"]).agg({"PRICE":"mean"}).sort_values(by = "PRICE",ascending = False)

df = df.reset_index()
df.head()

#["AGE_CUT"] variable creation

df["AGE_CUT"] = pd.cut(x= df["AGE"], bins=[0,15, 30, 45, 60, 75],labels=["0_15","16_30,","31_45","46_60","61_75"])

df.head()

#["customers_level_based"] variable creation

df["customers_level_based"] = df["COUNTRY"]+"_"+df["SOURCE"]+"_"+df["SEX"]+"_"+df["AGE_CUT"].astype(str)
df["customers_level_based"].head()

#Customer Segmentation & Classification

agg_df = agg_df.reset_index()
new_user = "tur_android_female_31_45"

agg_df[agg_df["customers_level_based"]== new_user]

new_user2 = "fra_ios_female_31_45"

agg_df[agg_df["customers_level_based"]== new_user2]
