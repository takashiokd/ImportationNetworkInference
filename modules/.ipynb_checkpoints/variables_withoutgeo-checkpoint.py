import geopandas as gpd
import pandas as pd
import numpy as np

dict_region_number_England={ 'North East':0,  'North West':1, 'Yorkshire and The Humber':2, 'East Midlands':3,'West Midlands':4,
     'East of England':5, 'London':6,'South East':7, 'South West':8}
# states_US=['Alabama',
#  'Alaska',
#  'Arizona',
#  'Arkansas',
#  'California',
#  'Colorado',
#  'Connecticut',
#  'Delaware',
#  'District of Columbia',
#  'Florida',
#  'Georgia',
#  'Hawaii',
#  'Idaho',
#  'Illinois',
#  'Indiana',
#  'Iowa',
#  'Kansas',
#  'Kentucky',
#  'Louisiana',
#  'Maine',
#  'Maryland',
#  'Massachusetts',
#  'Michigan',
#  'Minnesota',
#  'Mississippi',
#  'Missouri',
#  'Montana',
#  'Nebraska',
#  'Nevada',
#  'New Hampshire',
#  'New Jersey',
#  'New Mexico',
#  'New York',
#  'North Carolina',
#  'North Dakota',
#  'Ohio',
#  'Oklahoma',
#  'Oregon',
#  'Pennsylvania',
#  'Rhode Island',
#  'South Carolina',
#  'South Dakota',
#  'Tennessee',
#  'Texas',
#  'Utah',
#  'Vermont',
#  'Virginia',
#  'Washington',
#  'West Virginia',
#  'Wisconsin',
#  'Wyoming']

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


England_region_index =['North East',  'North West', 'Yorkshire and The Humber', 'East Midlands','West Midlands', 'East of England', 'London','South East', 'South West']

England_region_index_mod=['North East',  'North West', 'Yorkshire and\n The Humber', 'East Midlands','West Midlands', 'East of England', 'London','South East', 'South West']

England_region_index_abb = ['NE',  'NW', 'YH', 'EM','WM', 'EE', 'LDN','SE', 'SW']

dict_region_abb= dict({
    'North East':'NE',
 'North West':'NW',
 'Yorkshire and The Humber':'YH',
 'Yorkshire and\n The Humber':'YH',
 'East Midlands':'EM',
 'West Midlands':'WM',
 'East of England':'EE',
 'London':'LDN',
 'South East':'SE',
 'South West':'SW'
})

England_popden = dict({'North East':311,
 'North West':520,
 'Yorkshire and The Humber':357,
 'East Midlands':310,
 'West Midlands':457,
 'East of England':326,
 'London':5701,
 'South East':481,
 'South West':236})

# England_gdf = gpd.read_file('data/geography/Regions_(December_2017)_Boundaries.shp')
# # Re-order regions of gdf in the same order as the variable index. (For the above shapefile, this step can be omitted since the order is the same by default)
# reindex=[list(England_gdf['rgn17nm']).index(ind) for ind in England_region_index] 
# England_gdf=England_gdf.reindex(reindex)
# England_gdf=England_gdf.reset_index( drop=True)
# England_gdf=England_gdf.sort_index(axis=0,ascending=True)


# UTLA_gdf=gpd.read_file('data/geography/England_UTLA_administrative_areas.shp')



# # Create file of population sizes
aux=pd.read_csv('data/England_data/utla_popsize.csv')
df_pop_utla=pd.DataFrame() 
df_pop_utla['utla']= aux['Name']
df_pop_utla['pop']= aux['2020 Mid-year estimate']
not_in_popfile=pd.read_csv('data/England_data/not_in_popfile.csv',index_col=0)
not_in_popfile['pop']=[int(i) for i in not_in_popfile['pop']]
df_pop_utla=df_pop_utla.append(not_in_popfile)
df_pop_utla=df_pop_utla.reset_index(drop=True)
df_pop_utla['pop']=[int(str(i).replace(',','')) if str(i)!='No data' else np.nan for i in df_pop_utla['pop'] ]
#df_pop_utla.to_csv('data/England_data/utla_popsize_someUTLAadded.csv')




# Decompose US into 9 or 8 regions #https://www.nature.com/articles/s41598-021-90539-2

# # OLD VERSION
# US_regions = ['New England', 'Mid-Atlantic', 'South Atlantic','East South Central','West South Central','East North Central','West North Central','Mountains','Pacific' ]
# states_in_regions=[['Massachusetts', 'Connecticut', 'New Hampshire', 'Maine', 'Rhode Island','Vermont'],
#  ['New York', 'Pennsylvania', 'New Jersey', 'Maryland','Delaware','District of Columbia'],
#  ['Florida', 'Georgia', 'North Carolina','Virginia', 'South Carolina','West Virginia'],
#  ['Tennessee', 'Alabama', 'Kentucky','Mississippi'],
#  ['Texas', 'Louisiana', 'Oklahoma','Arkansas'],
#  ['Illinois', 'Ohio', 'Michigan', 'Indiana' , 'Wisconsin'],
# ['Missouri', 'Minnesota', 'Iowa', 'Kansas', 'Nebraska', 'South Dakota' ,'North Dakota'],
# ['Arizona', 'Colorado', 'Utah', 'Nevada', 'New Mexico', 'Idaho', 'Montana' ,'Wyoming'],
#  ['California', 'Washington', 'Oregon', 'Hawaii' ,'Alaska']
# ]

# # Combine East South Central and West South Central
US_regions = ['New England', 'Mid-Atlantic', 'South Atlantic','South Central','East North Central','West North Central','Mountains','Pacific' ]

states_in_regions=[['Massachusetts', 'Connecticut', 'New Hampshire', 'Maine', 'Rhode Island','Vermont'],
 ['New York', 'Pennsylvania', 'New Jersey', 'Maryland','Delaware','District of Columbia'],
 ['Florida', 'Georgia', 'North Carolina','Virginia', 'South Carolina','West Virginia'],
 ['Tennessee', 'Alabama', 'Kentucky','Mississippi','Texas', 'Louisiana', 'Oklahoma','Arkansas'],
 ['Illinois', 'Ohio', 'Michigan', 'Indiana' , 'Wisconsin'],
['Missouri', 'Minnesota', 'Iowa', 'Kansas', 'Nebraska', 'South Dakota' ,'North Dakota'],
['Arizona', 'Colorado', 'Utah', 'Nevada', 'New Mexico', 'Idaho', 'Montana' ,'Wyoming'],
 ['California', 'Washington', 'Oregon', 'Hawaii' ,'Alaska']
]


# states_ordered=[]
# for idx, i in enumerate(states_in_regions):
#     for j in i:
#         states_ordered.append([j,US_regions[idx]])
# df_states_US = pd.DataFrame(data=states_ordered,columns = ['state','region'])


# states_US = list(df_states_US['state'])
# US_gdf = gpd.read_file('data/geography/cb_2018_us_state_500k.shp')
# aux=[]
# for i in states_US:
#     aux.append(list(US_gdf['NAME']).index(i))
# US_gdf = US_gdf.iloc[aux]
# US_gdf=US_gdf.reset_index( drop=True)


US_adj = [
['New England','Mid-Atlantic'],
['Mid-Atlantic', 'South Atlantic'],
['Mid-Atlantic', 'East North Central'],
['East North Central', 'South Atlantic'],
['East North Central', 'South Central'],
['East North Central','West North Central'],
['South Atlantic', 'South Central'],
['South Central', 'West North Central'],
['West North Central', 'Mountains'],
['Mountains', 'South Central'],
['Mountains', 'Pacific'],
]
England_adj=[['North East', 'North West'],
 ['North East', 'Yorkshire and\n The Humber'],
 ['North West', 'Yorkshire and\n The Humber'],
 ['North West', 'East Midlands'],
 ['North West', 'West Midlands'],
 ['Yorkshire and\n The Humber', 'East Midlands'],
 ['East Midlands', 'West Midlands'],
 ['East Midlands', 'East of England'],
 ['East Midlands', 'South East'],
 ['West Midlands', 'South East'],
 ['West Midlands', 'South West'],
 ['East of England', 'London'],
 ['East of England', 'South East'],
 ['London', 'South East'],
 ['South East', 'South West']]


#from the Presidential Election 2020. Taken from https://www.gkgigs.com/list-of-blue-states-and-red-states/
US_blueD_redR = [['Arizona',
  'California',
  'Colorado',
  'Connecticut',
  'Delaware',
  'Georgia',
  'Hawaii',
  'Illinois',
  'Maine',
  'Maryland',
  'Massachusetts',
  'Michigan',
  'Minnesota',
  'Nevada',
  'New Hampshire',
  'New Jersey',
  'New Mexico',
  'New York',
  'Oregon',
  'Pennsylvania',
  'Rhode Island',
  'Vermont',
  'Virginia',
  'Washington',
  'Wisconsin'],
 ['Alabama',
  'Alaska',
  'Arkansas',
  'Florida',
  'Idaho',
  'Indiana',
  'Iowa',
  'Kansas',
  'Kentucky',
  'Louisiana',
  'Mississippi',
  'Missouri',
  'Montana',
  'Nebraska',
  'North Carolina',
  'North Dakota',
  'Ohio',
  'Oklahoma',
  'South Carolina',
  'South Dakota',
  'Tennessee',
  'Texas',
  'Utah',
  'West Virginia',
  'Wyoming']]



east_states = ['New England',
 'Mid-Atlantic',
 'South Atlantic',
 'East North Central']
west_states  = [ 'West North Central',
             'South Central',
 'Mountains',
 'Pacific']



#age_class=['0-19', '20-29', '30-39', '40-49', '50-59', '60-79']