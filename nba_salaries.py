import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from urllib.request import urlopen
from bs4 import BeautifulSoup

year = 2018
url = 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'.format(year)
salary_url = 'https://www.basketball-reference.com/contracts/players.html'
seed = 6
clusters = 10

not_yet_num_cols = ['Age','G','MP','PER','TS%','3PAr','FTr','ORB%','DRB%','TRB%',
                    'AST%','STL%','BLK%','TOV%','USG%','OWS','DWS','WS','WS/48',
                    'OBPM','DBPM','BPM','VORP']
df_interesting_variables = ['TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%',
                            'BLK%','TOV%','USG%','OWS','DWS','WS','WS/48','OBPM',
                            'DBPM','BPM', 'VORP']
# Part 1
html = urlopen(url)
soup = BeautifulSoup(html,"lxml")
cols = [th.getText() for th in soup.findAll('tr',limit=3)[0].findAll('th')]
cols.remove('Rk')

rows = soup.findAll('tr')[1:]
dat = [[td.getText() for td in rows[i].findAll('td')] for i in (range(len(rows)))]
df = pd.DataFrame(dat,columns=cols)
df = pd.concat([df[['Player','Pos','Tm',]], df[not_yet_num_cols].apply(pd.to_numeric, errors='ignore')], axis=1)

df = df[(df.G >= 50) & (df.MP >= 800) & (df.WS/48 > 0)]
df = df.reset_index()
df.to_csv('full_nba_data')

df_k_means = df[df_interesting_variables]
df_k_means.to_csv('clustering_nba_data')

corr = df_k_means.corr()
corr.style.background_gradient().set_precision(2)

scaler = StandardScaler()
kmeans = KMeans(n_clusters=clusters, random_state = seed)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit_transform(df_k_means)
labs = pipeline.predict(df_k_means)
results_df = pd.concat([df[['Player','Pos','Tm',]], df_k_means], axis=1)
results_df['label'] = labs.tolist()

salaries = pd.read_html(salary_url, header=1)[0]
salaries = salaries[salaries['Rk'] != 'Rk']
salaries = salaries[['Player', '2018-19', 'Guaranteed']]

for col in ['2018-19','Guaranteed']:
    salaries[col] = salaries[col].replace('[\$,]', '', regex=True).astype(float)

salaries.to_csv('player_salaries')

combined_df = pd.merge(results_df, salaries, how='left', left_on=['Player'], right_on=['Player'])
combined_df = combined_df.fillna(0)
combined_df.head()
combined_df.to_csv('joined_nba_data')

# Part 2
data = pd.read_csv('joined_nba_data', index_col = 0)
for i in list(range(0,10)):
    print("Group {} includes players such as {}\n".format(i, list(set(data[data.label==i]['Player'][0:10]))))

ax = sns.boxplot(x="label", y="2018-19",data=data)
plt.show()

ax = sns.regplot(x="WS/48", y="2018-19",data=data)
plt.show()

fig, ax = plt.subplots(nrows=4, ncols=3,figsize=(8, 4))
def plot_groups(axis, group):
    axis = sns.regplot(x="WS/48", y="2018-19",data=data[data.label==group])
    axis.set_title("Group {}".format(i))

for i in list(range(0,10)):
    axis = sns.regplot(x="WS/48", y="2018-19",data=data[data.label==i])
    axis.set_title("Group {}".format(i))
    plt.show()

analysis_df = data[['Player','Pos','Tm','label','2018-19']]
final_dict = {}

for clus in list(range(10)):
    small_df = analysis_df[analysis_df.label == clus]
    small_df.reset_index(drop=False,inplace=True)
    small_df.loc[:,'percentile'] = small_df['2018-19'].rank(pct=True)
    underpaid = small_df[small_df['percentile'] <= 0.25]['Player'].tolist()
    overpaid = small_df[small_df['percentile'] >= 0.75]['Player'].tolist()
    final_dict[clus] = [underpaid,overpaid]

for key, val in final_dict.items():
    print("In group {}, \nthe most underpaid players are {} and \nthe most overpaid players are {}\n".format(key, val[0], val[1]))
