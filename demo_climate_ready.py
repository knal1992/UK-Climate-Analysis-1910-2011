import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pylab
import glob
import os.path
from datetime import datetime
import dateutil.parser
from statsmodels.tsa.stattools import adfuller
from scipy import stats
#Geo data
from pyproj import Proj, transform
from pygeocoder import Geocoder
import reverse_geocoder as rg 


####### READ THE DATA INTO A PANDAS DATAFRAME AND SET THE CORRECT TIME SERIE AS INDEX, IT ALSO CREATES A CSV WITH THE CONCATENATED DATA   #####################
print ('####################     UK Climate Time Series Analysis     ####################')
#Modify the path according to the path where you have stored the csv files that you want to read
path = input('\nSpecify the path to files and end with / (for example /Users/krzysztofnalborski/Desktop/Climate_TS/meantemp/) : ')
path = str(path)

#User specifies
####################GLOBAL#################

VARIABLE = input('\nSpecify the variable: ')
VARIABLE = str(VARIABLE)

u = input('\nSpecify the unit of measurement, for example (Degrees Celcius): ')
u = str(u)

startDate = input('\nSpecify starting date of time series: for example 1910-01 or 1961-01 : ')
startDate = str(startDate)
endDate = '2012-01'

w = int(input('\nSpecify moving average window: for example 6 or 12: '))

win = str(w)

start = str(startDate[0:4])
end = str(endDate[0:4])

years = str(startDate[0:4]) + '-' + str(endDate[0:4]) 

season1 = 'Winters' + ' ' + str(startDate[0:4]) + '-' + '2011' 
season2 = 'Springs' + ' ' + str(startDate[0:4]) + '-' + '2011' 
season3 = 'Summers' + ' ' + str(startDate[0:4]) + '-' + '2011' 
season4 = 'Autumns' + ' ' + str(startDate[0:4]) + '-' + '2011'  

# for graphs (ranges) to be defined by user
print ('\nData for Comparison- Bar Charts')
a = input('Specify initial data range: for example 1910-1930: ')
x1 = str(a[0:4])
y1 = str(a[5:9])

b = input('Specify final data range: for example 1991-2011: ')
x2 = str(b[0:4])
y2 = str(b[5:9])

###########################################################

dates_serie = pd.date_range(startDate, endDate, freq='M')

#Check if the concatenated file with all the information already exists if so, it delete it
if os.path.exists(path + 'ConcatenatedDataTS.csv'):
    os.remove(path + 'ConcatenatedDataTS.csv')
    
files = glob.glob(path + '*.csv')
df = pd.DataFrame()

#This loop reads all the csv files in the folder traspose them (Dates become rows and coordinates become columns)
print ('\nAnalysis in Progress...')
for file in files:
    print(file)
    tmp_df = pd.read_csv(file)
    tmp_df = pd.DataFrame.transpose(tmp_df)
    df = df.append(tmp_df)

#Method imported from: https://webscraping.com/blog/Converting-UK-Easting-Northing-coordinates/
#Converting easting northing to latitude and longitude

v84 = Proj(proj="latlong",towgs84="0,0,0",ellps="WGS84")
v36 = Proj(proj="latlong", k=0.9996012717, ellps="airy",
        towgs84="446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894")
vgrid = Proj(init="world:bng")

def COORDINATES(easting, northing):
    vlon36, vlat36 = vgrid(easting, northing, inverse=True)
    return transform(v36, v84, vlat36, vlon36)
#############################################################################################################################  

#Extracting easting northing information from the data frame
easting = list(df.iloc[0])
northing  = list(df.iloc[1])

#Creating (easting, northing) tuple so it can be used in the COORDINATES method
easting_northing= list(zip(easting, northing))

#Converting easting, northing to latitude and longitude
lat_long = [COORDINATES(item[0], item[1]) for item in easting_northing]

#Just to add coordinate to data frame (just in case)
longitude = [value[1] for value in lat_long]
latitude = [value[0] for value in lat_long]

#List containing tuples of lat and long so country info can be extracted using rg.search below
coordinates = list(lat_long)

#Converting lat, long to country, town, etc.
#Method imported https://github.com/thampiman/reverse-geocoder/commit/2cf0b19398066362e086a58eadee1800e59d4120
country_details = rg.search(coordinates)

countries = []
towns = []
regions = []

#rg.search(coordinates) methods creates an ordered dictionary, the loop below extracts details regarding geolocation
for detail in country_details:
    country = detail['admin1']
    town = detail['name']
    region = detail['admin2']
    countries.append(country)
    towns.append(town)
    regions.append(region)

#Transponsing again just to add new columns at specific locations
df = pd.DataFrame.transpose(df)

df.insert(2, 'longitude', longitude)
df.insert(3, 'latitude', latitude)
df.insert(4, 'region', regions)
df.insert(5, 'town', towns)
df.insert(6, 'country', countries)

#Transponsing back to TimeSeries format
df = pd.DataFrame.transpose(df)

#Create a csv file with the concatenated information
df.to_csv(path + 'ConcatenatedDataTS.csv', header=False)

#Whole UK
whole_UK = df.copy()
whole_UK.drop(whole_UK.index[[0,1,2,3,4,5,6]], inplace=True)
whole_UK.set_index(dates_serie, inplace=True)
whole_UK_seasons = whole_UK.copy() #for seasons-related rolling stats
whole_UK_seasons = whole_UK_seasons.mean(axis=1) #for seasons-related rolling stats

#Used to extract specific countries/towns/regions
UK_countries = df.copy()
cols_countries = UK_countries.iloc[6]
#print (cols_countries)
cols_countries= list(cols_countries)
UK_countries.drop(UK_countries.index[[0,1,2,3,4,5,6]], inplace=True)
UK_countries.columns = cols_countries #now each column is renamed representing country of 5x5 grid

#Data for Scotland
Scotland = UK_countries['Scotland']
Scotland.set_index(dates_serie, inplace=True)
Scotland_seasons = Scotland.copy()
Scotland_seasons = Scotland_seasons.mean(axis=1)

#Data for England
England = UK_countries['England']
England.set_index(dates_serie, inplace=True)
England_seasons = England.copy()
England_seasons = England_seasons.mean(axis=1)

#Data for Wales
Wales = UK_countries['Wales']
Wales.set_index(dates_serie, inplace=True)
Wales_seasons = Wales.copy()
Wales_seasons = Wales_seasons.mean(axis=1)

#Data for Northern Ireland
NI = UK_countries['Northern Ireland']
NI.set_index(dates_serie, inplace=True)
NI_seasons = NI.copy()
NI_seasons = NI_seasons.mean(axis=1)

#############################################################################################################################
"""All countries together- annual averages- rolling stats"""
#Dickey-Fuller method imported from: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/ 
def rollings_stats_all_annual(w, start, end, var_name, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, df_seasons1, df_seasons2, df_seasons3, df_seasons4, df_seasons5, season_name):
    #Getting annual averages
    season_time = pd.date_range(start, end, freq= 'A') #only used for plotting
    season_months = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]
    
    #Whole Country
    season1 = df_seasons1[df_seasons1.index.map(lambda t: t.month in season_months)]
    season1 = season1.groupby(np.arange(len(season1))//12).mean()
    
    #Country 1
    season2 = df_seasons2[df_seasons2.index.map(lambda t: t.month in season_months)]
    season2 = season2.groupby(np.arange(len(season2))//12).mean()
    
    #Country 2
    season3 = df_seasons3[df_seasons3.index.map(lambda t: t.month in season_months)]
    season3 = season3.groupby(np.arange(len(season3))//12).mean()
    
    #Country 3
    season4 = df_seasons4[df_seasons4.index.map(lambda t: t.month in season_months)]
    season4 = season4.groupby(np.arange(len(season4))//12).mean()
    
    #Country 4
    season5 = df_seasons5[df_seasons5.index.map(lambda t: t.month in season_months)]
    season5 = season5.groupby(np.arange(len(season5))//12).mean()
    
     #Stats
    rolmean1 = pd.Series(season1).rolling(window=w).mean()
    rolmean2 = pd.Series(season2).rolling(window=w).mean()
    rolmean3 = pd.Series(season3).rolling(window=w).mean()
    rolmean4 = pd.Series(season4).rolling(window=w).mean()
    rolmean5 = pd.Series(season5).rolling(window=w).mean()
    
    #Perform Dickey-Fuller test1:
    print ('\n***UK' + '-' + 'Results of Dickey-Fuller Test:***')
    dftest = adfuller(df_seasons1, autolag='AIC')
    p_value =  dftest[1]
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if p_value > 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is NOT stationary')
        print ('Alpha Level = 0.05')
    elif p_value < 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is stationary')
        print ('Alpha Level = 0.05')
    
    #Perform Dickey-Fuller test2:
    print ('\n***Scotland' + '-' + 'Results of Dickey-Fuller Test:***')
    dftest = adfuller(df_seasons2, autolag='AIC')
    p_value =  dftest[1]
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if p_value > 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is NOT stationary')
        print ('Alpha Level = 0.05')
    elif p_value < 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is stationary')
        print ('Alpha Level = 0.05')
    
    #Perform Dickey-Fuller test3:
    print ('\n***Enland' + '-' + 'Results of Dickey-Fuller Test: ***')
    dftest = adfuller(df_seasons3, autolag='AIC')
    p_value =  dftest[1]
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if p_value > 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is NOT stationary')
        print ('Alpha Level = 0.05')
    elif p_value < 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is stationary')
        print ('Alpha Level = 0.05')
    
    #Perform Dickey-Fuller test4:
    print ('\n***Wales' + '-' + 'Results of Dickey-Fuller Test: ***')
    dftest = adfuller(df_seasons4, autolag='AIC')
    p_value =  dftest[1]
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if p_value > 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is NOT stationary')
        print ('Alpha Level = 0.05')
    elif p_value < 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is stationary')
        print ('Alpha Level = 0.05')
   
    
    #Perform Dickey-Fuller test5:
    print ('\n***Northern Ireland' + '-' + 'Results of Dickey-Fuller Test: ***')
    dftest = adfuller(df_seasons5, autolag='AIC')
    p_value =  dftest[1]
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if p_value > 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is NOT stationary')
        print ('Alpha Level = 0.05')
    elif p_value < 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is stationary')
        print ('Alpha Level = 0.05')

    #Plots
    mean1 = plt.plot(season_time, rolmean1, color='blue', label='UK')
    mean2 = plt.plot(season_time, rolmean2, color='red', label='Scotland')
    mean3 = plt.plot(season_time, rolmean3, color='green', label='England')
    mean4 = plt.plot(season_time, rolmean4, color='black', label='Wales')
    mean5 = plt.plot(season_time, rolmean5, color='orange', label='Northern Ireland')
    
    plt.ylabel(u)
    pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.07), ncol=5)
    plt.title(season_name + ' ' + x1 + '-' + y2 + ' ' + var_name + ' ' + '(' + win + '-' + 'year Rolling Mean)')
    plt.grid()
    print ('\n!!!   Close Figure 1 to continue analysis   !!!')
    plt.show()
    

print ('\n##########     Statistics and Visualizations     ##########')
print ('\n' + 'See Figure 1' + ' ' + '(' + 'Mean Annual' + ' ' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'for vizualization' +')')
#Calling the method
rollings_stats_all_annual(w, start, end, VARIABLE,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, whole_UK_seasons, \
Scotland_seasons, England_seasons, Wales_seasons, NI_seasons, 'Average Annual')


############################################################################################################################
"""Method for obtaining rolling stats for seasons in UK"""
def rollings_stats_seasons(w, var_name, m1, m2, m3, df_seasons, season_name, country_name):
    #Getting seasons (winter, spring, summer, autumn)
    season_months = [m1, m2, m3]
    season = df_seasons[df_seasons.index.map(lambda t: t.month in season_months)]
    season = season.groupby(np.arange(len(season))//3).mean()
    
    #Stats
    rolmean = pd.Series(season).rolling(window=w).mean()
    
    #Perform Dickey-Fuller test:
    print ('\n***',season_name, ': UK- Results of Dickey-Fuller Test: ***')
    dftest = adfuller(df_seasons, autolag='AIC')
    p_value =  dftest[1]
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if p_value > 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is NOT stationary')
        print ('Alpha Level = 0.05')
    elif p_value < 0.05:
        print ('\nInterpretation:')
        print ('Based on p-value data is stationary')
        print ('Alpha Level = 0.05')
    
    
    return rolmean

print ('\n##########     Statistics and Visualizations for seasons in the UK     ##########')
print ('\n' + 'See Figure 1' + ' ' + '(' + 'UK' + ' ' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'by seasons' + ' ' + 'for vizualization' +')')
#Rolling stats seasons for the whole UK
winUK = rollings_stats_seasons(w, VARIABLE, 12, 1, 2, whole_UK_seasons, season1, 'UK')
sprUK = rollings_stats_seasons(w, VARIABLE, 3, 4, 5, whole_UK_seasons, season2, 'UK')
sumUK = rollings_stats_seasons(w, VARIABLE, 6, 7, 8, whole_UK_seasons, season3, 'UK')
autUK = rollings_stats_seasons(w, VARIABLE, 9, 10, 11, whole_UK_seasons, season4, 'UK')

#Plots
season_time = pd.date_range(start, end, freq= 'A') #only used for plotting
mean1 = plt.plot(season_time, winUK, color='blue', label='Winter')
mean2 = plt.plot(season_time, sprUK, color='green', label='Spring')
mean3 = plt.plot(season_time, sumUK, color='red', label='Summer')
mean4 = plt.plot(season_time, autUK, color='black', label='Autumn')
    
pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.07), ncol=5)
plt.title('UK' + ' ' + VARIABLE + ' ' + x1 + '-' + y2 + ' ' + 'by seasons' + ' ' + '(' + win + '-' + 'year Rolling Mean)')
plt.grid()
print ('\n!!!   Close Figure 1 to continue analysis   !!!')
plt.show()

##############################################################################################################################
"""Grouped Graph for Annual Values"""
UK1 = whole_UK_seasons.ix[x1 + '-01-31':y1 + '-12-31']
uk1 = UK1.mean(axis=0)

UK2 = whole_UK_seasons.ix[x2 + '-01-31':y2 + '-12-31']
uk2 = UK2.mean(axis=0)

ttest1 = stats.ttest_rel(UK1, UK2)
p1 = ttest1[1]
print ('\n##########     Comparing the means     ##########')
print ('See Figure 1' + ' ' + '(' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'for vizualization' +')')
print ('\n***T-test Results for UK: ***')
ttoutput = pd.Series(ttest1[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p1 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p1 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#
Scot1 = Scotland_seasons.ix[x1 + '-01-31':y1 + '-12-31']
scot1 = Scot1.mean(axis=0)

Scot2 = Scotland_seasons.ix[x2 + '-01-31':y2 + '-12-31']
scot2 = Scot2.mean(axis=0)

ttest2 = stats.ttest_rel(Scot1, Scot2)
p2 = ttest2[1]
print ('\n***T-test Results for Scotland: ***')
print ('T-test Results for UK:')
ttoutput = pd.Series(ttest2[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p2 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p2 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#
Eng1 = England_seasons.ix[x1 + '-01-31':y1 + '-12-31']
eng1 = Eng1.mean(axis=0)

Eng2 = England_seasons.ix[x2 + '-01-31':y2 + '-12-31']
eng2 = Eng2.mean(axis=0)

ttest3 = stats.ttest_rel(Eng1, Eng2)
p3 = ttest3[1]
print ('\n***T-test Results for England: ***')
ttoutput = pd.Series(ttest3[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p3 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p3 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#
Wal1 = Wales_seasons.ix[x1 + '-01-31':y1 + '-12-31']
wal1 = Wal1.mean(axis=0)

Wal2 = Wales_seasons.ix[x2 + '-01-31':y2 + '-12-31']
wal2 = Wal2.mean(axis=0)

ttest4 = stats.ttest_rel(Wal1, Wal2)
p4 = ttest4[1]
print ('\n***T-test Results for Wales: ***')
ttoutput = pd.Series(ttest4[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p4 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p4 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#
NI1 = NI_seasons.ix[x1 + '-01-31':y1 + '-12-31']
ni1 = NI1.mean(axis=0)

NI2 = NI_seasons.ix[x2 + '-01-31':y2 + '-12-31']
ni2 = NI2.mean(axis=0)

ttest5 = stats.ttest_rel(NI1, NI2)
p5 = ttest5[1]
print ('\n***T-test Results for Northern Ireland: ***')
ttoutput = pd.Series(ttest5[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p5 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value')  
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p5 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#Graphs
N = 5
means11 = (uk1, scot1, eng1, wal1, ni1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects11 = ax.bar(ind, means11, width, color='orange', label= x1 + '-' + y1)

means22= (uk2, scot2, eng2, wal2, ni2)
rects22 = ax.bar(ind + width, means22, width, color='green', label= x2 + '-' + y2)

Percentage_diff = list(zip(means11, means22))
diff = []
for i in Percentage_diff:
    p = ((i[1]*100)/i[0])-100
    p = '%.2f%%' % p
    diff.append(p)

print ('\n% change' + ' ' + 'in' + ' ' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 )
per1 = pd.Series(diff[0:5], index=['UK','Scotland', 'England', 'Wales', 'Northern Ireland'])
print (per1)

print ('\n' + 'See Figure 1' + ' ' + '(' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'for vizualization' +')')

# add some text for labels, title and axes ticks
ax.set_ylabel(u)
ax.set_title(VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('UK', 'Scotland', 'England', 'Wales', 'Northern Ireland'))

pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.07), ncol=5)
plt.tight_layout
plt.grid()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.90*height,
                '%.2f' % (height),
                ha='center', va='bottom')

autolabel(rects11)
autolabel(rects22)
print ('!!!   Close Figure 1 to continue analysis   !!!')
plt.show()



############################################################################################################################
#Getting data for seasons for the whole UK
print ('\n##########     Comparing the means by seasons     ##########')
print ('See Figure 1' + ' ' + '(' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'by seasons' + ' ' + 'for vizualization' +')')
#Winter
UKWjan = whole_UK_seasons.ix[x1 + '-01-31':y1 + '-01-31':12]
UKWfeb = whole_UK_seasons.ix[x1 + '-02-28':y1 + '-02-28':12]
UKWdec = whole_UK_seasons.ix[x1 + '-12-31':y1 + '-12-31':12]
Winter = list(zip(UKWjan, UKWfeb, UKWdec))
W1= [] #for t-test
for i in Winter:
    x = sum(i)/len(i)
    W1.append(x)

Winters_UK1 = sum(W1)/len(W1)

UKWjan2 = whole_UK_seasons.ix[x2 + '-01-31':y2 + '-01-31':12]
UKWfeb2 = whole_UK_seasons.ix[x2 + '-02-28':y2 + '-02-28':12]
UKWdec2 = whole_UK_seasons.ix[x2 + '-12-31':y2 + '-12-31':12]
Winter2 = list(zip(UKWjan2, UKWfeb2, UKWdec2))
W2 = [] # for t-test
for i in Winter2:
    x = sum(i)/len(i)
    W2.append(x)

Winters_UK2 = sum(W2)/len(W2)

ttestwinter = stats.ttest_rel(W1, W2)
p11 = ttestwinter[1]
print ('\n***T-test Results for Winters: ***')
ttoutput = pd.Series(ttestwinter[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p11 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p11 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#Spring
UKSmar = whole_UK_seasons.ix[x1 + '-03-31':y1 + '-03-31':12]
UKSapr = whole_UK_seasons.ix[x1 + '-04-30':y1 + '-04-30':12]
UKSmay = whole_UK_seasons.ix[x1 + '-05-31':y1 + '-05-31':12]
Spring = list(zip(UKSmar, UKSapr, UKSmay))
S1 = [] #for t-test
for i in Spring:
    x = sum(i)/len(i)
    S1.append(x)

Springs_UK1 = sum(S1)/len(S1)

UKSmar2 = whole_UK_seasons.ix[x2 + '-03-31':y2 + '-03-31':12]
UKSapr2 = whole_UK_seasons.ix[x2 + '-04-30':y2 + '-04-30':12]
UKSmay2 = whole_UK_seasons.ix[x2 + '-05-31':y2 + '-05-31':12]
Spring2 = list(zip(UKSmar2, UKSapr2, UKSmay2))
S2 = [] #for t-test
for i in Spring2:
    x = sum(i)/len(i)
    S2.append(x)

Springs_UK2 = sum(S2)/len(S2)

ttestspring = stats.ttest_rel(S1, S2)
p22 = ttestspring[1]
print ('\n***T-test Results for Springs: ***')
ttoutput = pd.Series(ttestspring[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p22 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value')  
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p22 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#Summer
UKSUjun = whole_UK_seasons.ix[x1 + '-06-30':y1 + '-06-30':12]
UKSUjul = whole_UK_seasons.ix[x1 + '-07-31':y1 + '-07-31':12]
UKSUaug = whole_UK_seasons.ix[x1 + '-08-31':y1 + '-08-31':12]
Summer = list(zip(UKSUjun, UKSUjul, UKSUaug))
Su1 = [] #for t-test
for i in Summer:
    x = sum(i)/len(i)
    Su1.append(x)

Summers_UK1 = sum(Su1)/len(Su1)

UKSUjun2 = whole_UK_seasons.ix[x2 + '-06-30':y2 + '-06-30':12]
UKSUjul2 = whole_UK_seasons.ix[x2 + '-07-31':y2 + '-07-31':12]
UKSUaug2 = whole_UK_seasons.ix[x2 + '-08-31':y2+ '-08-31':12]
Summer2 = list(zip(UKSUjun2, UKSUjul2, UKSUaug2))
Su2 = [] #for t-test
for i in Summer2:
    x = sum(i)/len(i)
    Su2.append(x)

Summers_UK2 = sum(Su2)/len(Su2)

ttestsummer = stats.ttest_rel(Su1, Su2)
p33 = ttestsummer[1]
print ('\n***T-test Results for Summers: ***')
ttoutput = pd.Series(ttestsummer[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p33 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p33 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#Autumn
UKAsep = whole_UK_seasons.ix[x1 + '-09-30':y1 + '-09-30':12]
UKAoct = whole_UK_seasons.ix[x1 + '-10-31':y1 + '-10-31':12]
UKAnov = whole_UK_seasons.ix[x1 + '-11-30':y1 + '-11-30':12]
Autumn = list(zip(UKAsep, UKAoct, UKAnov))
A1 = [] #for t-test
for i in Autumn:
    x = sum(i)/len(i)
    A1.append(x)

Autumns_UK1 = sum(A1)/len(A1)

UKAsep2 = whole_UK_seasons.ix[x2 + '-09-30':y2 + '-09-30':12]
UKAoct2 = whole_UK_seasons.ix[x2 + '-10-31':y2 + '-10-31':12]
UKAnov2 = whole_UK_seasons.ix[x2 + '-11-30':y2 + '-11-30':12]
Autumn2 = list(zip(UKAsep2, UKAoct2, UKAnov2))
A2 = [] #for t-test
for i in Autumn2:
    x = sum(i)/len(i)
    A2.append(x)

Autumns_UK2 = sum(A2)/len(A2)

ttestautumn = stats.ttest_rel(A1, A2)
p44 = ttestautumn[1]
print ('\n***T-test Results for Autumns: ***')
ttoutput = pd.Series(ttestautumn[0:2], index=['Test Statistic','p-value'])
print (ttoutput)
if p44 < 0.05:
    print ('\nInterpretation:')
    print ('Significantly Different mean values based on p-value') 
    print ('Variable:', VARIABLE)
    print ('Alpha Level = 0.05')
elif p44 > 0.05:
    print ('\nInterpretation:')
    print ('Variable:', VARIABLE)
    print ('Difference not statistically significant')
    print ('Alpha Level = 0.05')

#Graphs
N = 4
means1 = (Winters_UK1, Springs_UK1, Summers_UK1, Autumns_UK1)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, means1, width, color='orange', label= x1 + '-' + y1)

means2= (Winters_UK2, Springs_UK2, Summers_UK2, Autumns_UK2)
rects2 = ax.bar(ind + width, means2, width, color='green', label= x2 + '-' + y2)

Percentage_diff2 = list(zip(means1, means2))
diff2 = []
for i in Percentage_diff2:
    p2 = ((i[1]*100)/i[0])-100
    p2 = '%.2f%%' % p2
    diff2.append(p2)
    
print ('\n% change' + ' ' + 'in' + ' ' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 )
per1 = pd.Series(diff2[0:4], index=['Winters','Springs', 'Summers', 'Autums'])
print (per1)

print ('\n' + 'See Figure 1' + ' ' + '(' + VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'by seasons' + ' ' + 'for vizualization' +')')

# add some text for labels, title and axes ticks
ax.set_ylabel(u)
ax.set_title(VARIABLE + ' ' + x1 + '-' + y1 + ' ' + 'vs.' + ' ' + x2 + '-' + y2 + ' ' + 'by seasons')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Winter', 'Spring', 'Summer', 'Autumn'))

pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.07), ncol=5)
plt.tight_layout
plt.grid()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.90*height,
                '%.2f' % (height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.show()
