from cProfile import label
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


df_list=list()
result_list=list()

dir_content = os.listdir('results')
if '.DS_Store' in dir_content:
        print('REMOVED: .DS_Store')
        dir_content.remove('.DS_Store')

for result in dir_content:
    # if result == '.DS_Store':
    #     os.remove(os.path.join('results', result))
    df_=pd.read_csv(os.path.join('results',result,'reward.csv'), usecols=['reward'])
    df_.rename(columns={'reward':result},inplace=True)
    df_list.append(df_)
    result_list.append(result)
result_df=pd.concat(df_list, axis=1)
result_list.sort()
result_df = result_df.sort_index(axis=1)

# print(f'Best absout result:\n{result_df.max()}')
# print(f'Best rolling mean result:\n{result_df.rolling(window=100).mean().max()}')

#Show reward plot
fig = go.Figure()
for result in result_list:
    fig.add_trace(go.Scatter(
    x=result_df.index,
    y=result_df[result],
    name=result))
fig.update_layout(xaxis_title='Episode', yaxis_title='Reward')
fig.write_html(os.path.join('plots','reward_plot.html'))
# fig.show()

#Show log-y plot
fig = go.Figure()
for result in result_list:
    fig.add_trace(go.Scatter(
    x=result_df.index,
    y=result_df[result]*-1,
    name=result))
fig.update_layout(xaxis_title='Episode', yaxis_title='Reward log-scale')
fig.update_yaxes(type="log")
fig.write_html(os.path.join('plots','reward_logplot.html'))

# fig.show()

mov_avg_window = 10 # 100
boundary_range = 250

#Show running mean plot
fig = go.Figure()
for result in result_list:
    fig.add_trace(go.Scatter(
    x=result_df.index,
    y=result_df[result].rolling(window=mov_avg_window).mean(),
    name=result))
fig.update_layout(xaxis_title='Episode', yaxis_title=f'Mean Reward (Running {mov_avg_window})')
fig.write_html(os.path.join('plots','meanreward_plot.html'))
fig.show()

result_pct_diff = []

for result in range(len(result_list)):
    for res in range(len(result_list) - 1):
        result_0 = result_df[result_list[result]].rolling(window=mov_avg_window).mean().iloc[-1]
        result_1 = result_df[result_list[res]].rolling(window=mov_avg_window).mean().iloc[-1]
        
        result_pct = ((result_0 - result_1) / ((result_0 + result_1) / 2)) * 100
        result_pct_diff.append({f'{result_list[result]} / {result_list[res]}': '{:.3f}'.format(result_pct)})
print(*result_pct_diff, sep='\n')


print(f'Best rolling mean result:\n{result_df.tail(n=2000).rolling(window=mov_avg_window).mean().max()}')
# # print(f'Best index:\n{result_df.rolling(window=mov_avg_window).mean().idxmax()}')
print(f'Lower rolling mean ({boundary_range} eps) result:\n{result_df.tail(n=boundary_range).rolling(window=mov_avg_window).mean().min()}')
# # print(f'Lower index ({boundary_range} eps) result:\n{result_df.tail(n=boundary_range).rolling(window=mov_avg_window).mean().idxmin()}')
print(f'Upper rolling mean ({boundary_range} eps) result:\n{result_df.tail(n=boundary_range).rolling(window=mov_avg_window).mean().max()}')
# # print(f'Upper index ({boundary_range} eps) result:\n{result_df.tail(n=boundary_range).rolling(window=mov_avg_window).mean().idxmax()}')

#Show running mean y-log plot
# colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
# colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
colors = ['#42d4f4', '#911eb4', '#e6194B', '#ffa600', '#3cb44b', '#003f5c', ]
#  955196
# customPalette = sns.set_palette(sns.color_palette(colors))

fig = go.Figure()
matplot_fig = plt.figure()
ax = matplot_fig.add_subplot(111)
for result_id, result in enumerate(result_list):
    fig.add_trace(go.Scatter(
    x=result_df.index,
    y=result_df[result].rolling(window=mov_avg_window).mean()*-1,
    name=result))
    plt.plot(result_df.index, result_df[result].rolling(window=mov_avg_window).mean()*-1, label=result)
    # plt.plot(result_df.index, result_df[result].rolling(window=mov_avg_window).mean()*-1, label=result, color=colors[result_id])
    # sns.lineplot(data=result_df[result].rolling(window=mov_avg_window).mean()*-1, palette=customPalette, err_style='band')
    plt_mean = result_df[result].rolling(window=mov_avg_window).mean()*-1
    plt_std = result_df[result].rolling(window=mov_avg_window).std(ddof=0)*-1
    # ax.errorbar(result_df.index, plt_mean, yerr=plt_std, alpha=0.1)
    # plt.fill_between(x=result_df.index, y1=plt_mean - plt_std, y2=plt_mean + plt_std, alpha=0.2)
fig.update_layout(xaxis_title='Episode', yaxis_title=f'Mean Reward (Running {mov_avg_window}) log-scale')
fig.update_yaxes(type="log")
fig.write_html(os.path.join('plots','meanreward_logplot.html'))
# fig.show()
plt.title('3 Intersections')
plt.xlabel('Episode')
plt.ylabel(f'Mean Reward (Running {mov_avg_window}) log-scale')
plt.yscale('log')
# h = plt.gca().get_lines()
# print(h)
# plt.legend(handles=h, labels=result_list, loc='best')
# plt.legend()
plt.show()
# plt.savefig(os.path.join(path if config['is_train'] else plot_path, 'training_reward.png'))
