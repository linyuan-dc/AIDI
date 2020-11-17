# Here is the abstract of assignment 2 and final project.
#
#     Write a python module to periodically scan and parse news from newswire website. (20 Marks)
#     https://www.prnewswire.com/news-releases/news-releases-list/
#     Keep track of the already parsed news:  (20 Marks)
#     For all unparsed news, scan the content of the news to find a stock symbol.
#     e.g. (TSX:SHOP)
#     Scan the yahoo finance for the stock symbol appeared in the news. (30 Marks)
#     - Get the stock price and volume of last 5 days.
#     Prepare a nice visualization showing the News headline and Stock prices of last 5 days. (30 Marks)
#
# - visualization should be a plot (time - series) for
# - Volume
# - Daily Close Price

import sys 
import pandas as pd
import numpy as np 

from lxml.html import parse
from urllib.request import urlopen, Request
from pandas.io.parsers import TextParser

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from datetime import datetime
from importlib import import_module
import plotly.graph_objects as go

def _geturl(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0",
        }
    return urlopen(Request(url, headers=headers))

def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]

def _parse_options_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[1:]]
    #display(data)
    header =  [x.replace('*','') for x in header]
    return TextParser(data, names=header).get_chunk()

# get symbol quotes from Yahoo Finance
def get_quotes(symbol, num=5):
    url = f'https://finance.yahoo.com/quote/{symbol}/history'
    page = _geturl(url)
    parsed = parse(page)
    doc = parsed.getroot()
    tables = doc.findall('.//table')
    df = _parse_options_data(tables[0])
    # symbol is not in yahoo finance db, log and return
    if 'Date' not in df.columns:
        print(f"Can not retreive quote for symbol {symbol} by url {url}")
        print(df)
        return pd.DataFrame()
    # only keep top {num} rows
    df=df[0:min(len(df),num)]
    # data format clean 
    for x in df.columns:
        if x=='Date':
            df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        elif df[x].dtype!=np.float64:
            df[x] = df[x].str.replace(',','').astype(float) 
    return df.sort_values('Date').reset_index(drop=True)


# get news list from www.prnewswire.com
def getnews(url = 'https://www.prnewswire.com/news-releases/news-releases-list/'):
    page = _geturl(url)
    parsed = parse(page)
    doc = parsed.getroot()
    newslist = []
    for lnk in doc.findall(".//div[@class='card']"):
        news = {
            'thumbnail': lnk.find(".//img").get('src'),
            'time_short':     lnk.find(".//small").text_content(),
            'link':     'https://www.prnewswire.com' + lnk.find(".//a").get('href'),
            'title':    lnk.find(".//a[@title]").text_content(),
            'abstract': lnk.find(".//p").text_content(),
        }
        # get news content
        content_doc = parse(_geturl(news['link'])).getroot()
        news['time'] = content_doc.find(".//p[@class='mb-no']").text_content()
        news['content'] = content_doc.xpath(".//section[contains(@class, 'release-body')]")[0].text_content()
        newslist.append(news)
    return pd.DataFrame(newslist)

# use regular expression to get the stock symbols in news 
def extract_symbols(content):
    import re
    symbols = []
    for ex in ['TSX','NYSE','AMEX','Nasdaq']:
        re_pattern = rf'\b{ex}\s*\:\s*\b(\w+)\b'
        groups = re.findall(re_pattern, content, re.I)
        # add .TO to those symbols in TSX, as Yahoo Finance treat TSX symbols this way
        if ex=='TSX':
            tickers = [ (f'TSX:{x}', x + '.TO') for x in groups]
        else:
            tickers = [ (f'{ex}:{x}', x ) for x in groups]
        symbols.extend(tickers)
    return symbols

def showpic(url):
    im_data = plt.imread( _geturl(url), 0)
    height, width, depth = im_data.shape

    dpi = 72
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    im = ax.imshow(im_data, cmap='gray')
    plt.show()

class CandleStickChart():
    def __init__(self):
        # If mplfinance moudle installed, use mplfinance for Candle Stick Chart.
        # If not, use plotly to draw the chart.
        try:
            module = import_module('mplfinance.original_flavor')
            self.candlestick_api = getattr(module, 'candlestick2_ohlc')
            self.use_plotly = False
        except:
            self.use_plotly = True
    
    def show(self, df, stock_name):
        if self.use_plotly:
            return self.draw_candlestick(df,stock_name)
        else:
            return self.draw_candlestick_mpl(df,stock_name,self.candlestick_api)

    def draw_candlestick(self,df,stock_name):
        layout = go.Layout(title_text=stock_name,title_font_size=18, autosize=False, margin=go.layout.Margin(l=10, r=1, b=10),
                        xaxis=dict(title_text="Date", type='category',rangeslider= {'visible': False}),
                        yaxis=dict(title_text="<b>Price</b>", domain = [0.3, 1]),
                        yaxis2 = dict(title_text="<b>Volume</b>", anchor="x", domain = [0, 0.2]),
                        width=500, height=600  )
        
        candle = go.Candlestick(x=df['Date'].dt.date,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], increasing_line_color='#7bc0a3',
                        decreasing_line_color='#f6416c', name="Price")
        # Set volume bar color
        df['diag'] = '#7bc0a3' 
        df.loc[df['Close'] <= df['Open'] , 'diag'] = '#f6416c' 

        vol = go.Bar(x=df['Date'].dt.date, width=0.5,
                    y=df['Volume'], name="Volume", marker_color=df['diag'], opacity=0.8, yaxis='y2')  
        data = [candle, vol]
        fig = go.Figure(data, layout)
        fig.update(layout_showlegend=False)
        fig.show()

    #from mplfinance.original_flavor import candlestick2_ohlc
    def draw_candlestick_mpl(self,df,stock_name,candlestick2_ohlc_api):
        fig = plt.figure(figsize=(7,6), facecolor="white") 
        gs = gridspec.GridSpec(2, 1, left=0.08, bottom=0.15, right=0.99, top=0.96, wspace=0.2, hspace=0, height_ratios=[3.5,1])
        graph_KAV = fig.add_subplot(gs[0,:])
        graph_VOL = fig.add_subplot(gs[1,:])
        # draw candle chart
        
        candlestick2_ohlc_api(graph_KAV, df['Open'], df['High'], df['Low'], df['Close'], width=0.3,
                            colorup='g', colordown='r')  

        graph_KAV.grid(which='major', color='#CCCCCC', linestyle=':')
        graph_KAV.set_title(stock_name)
        graph_KAV.set_ylabel(u"Price")
        graph_KAV.set_xlim(-1, len(df.index)) 
        # draw bar chart for volume
        graph_VOL.bar(np.arange(0, len(df.index)), df.Volume, width=0.3, color=['r' if df.Open[x] > df.Close[x] else 'g' for x in range(0,len(df.index))])
        graph_VOL.set_ylabel(u"Volume")
        graph_VOL.set_xlim(-1,len(df.index)) 
        graph_VOL.set_xticklabels([''] + list(df['Date'].dt.date))

        for label in graph_KAV.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in graph_VOL.xaxis.get_ticklabels():
            label.set_rotation(45)
            label.set_fontsize(10)  
        plt.show()

def pollonce(df_old = pd.DataFrame()):
    df_new = getnews()
    if len(df_old) > 0:
        df_new = df_new[~df_new['link'].isin(df_old['link'])]
    for index, row in df_new.iterrows():
        print(f"\n\n{row['time']}:\n{row['title']}\n")
        showpic(row['thumbnail'])
        for name,symbol in extract_symbols(row['content']):
            #print(f'name:{name} , symbol:{symbol}')
            df_quotes = get_quotes(symbol)
            if len(df_quotes) > 0:
               CandleStickChart().show(df_quotes,name)
    return df_old.append(df_new, ignore_index=True)

def pollnews(interval = 10):
    import time
    df = pollonce()
    while True:
        print(f'wait for {interval} seconds')
        time.sleep( interval )
        print(f'Check news at {time.ctime()}')
        df = pollonce(df)


def main(argv):
    print("For debug")
    df = get_quotes('VTIQ')
    print(df)
    for name,symbol in extract_symbols('dsfsd lsd (TSX:SHOP) dsfsdf'):
        print(f'name:{name} , symbol:{symbol}')
        df_quotes = get_quotes(symbol)
        if len(df_quotes) > 0:
            CandleStickChart().show(df_quotes,name)
            #draw_candlestick(df_quotes,name)
    #pollnews()

if __name__ == "__main__":
   main(sys.argv[0:])