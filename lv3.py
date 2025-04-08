import urllib.request
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def fetch_air_quality_data():
    url = 'http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=0&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'
    airQualityHR = urllib.request.urlopen(url).read()
    return ET.fromstring(airQualityHR)

def process_data(root):
    df = pd.DataFrame(columns=('mjerenje', 'vrijeme'))
    
    children = list(root)
    for i, child in enumerate(children):
        obj = list(child)
        row = dict(zip(['mjerenje', 'vrijeme'], [obj[0].text, obj[2].text]))
        row_s = pd.Series(row)
        row_s.name = i
        df = pd.concat([df, row_s])
        df.at[i, 'mjerenje'] = float(df.at[i, 'mjerenje'])
    
    df['vrijeme'] = pd.to_datetime(df['vrijeme'], utc=True)
    df['month'] = df['vrijeme'].dt.month
    df['dayOfweek'] = df['vrijeme'].dt.dayofweek
    return df

def plot_data(df):
    df.plot(y='mjerenje', x='vrijeme')
    plt.show()

def get_top_3_days(df):
    return df.sort_values(['mjerenje'], ascending=False).head(3)

if __name__ == "__main__":
    root = fetch_air_quality_data()
    df = process_data(root)
    plot_data(df)
    print("Top 3 dana s najvećom koncentracijom PM10:")
    print(get_top_3_days(df))