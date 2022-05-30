print('running')
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
#pd.set_option('float_format', '{:f}'.format)
pd.options.display.float_format = '{:.2f}'.format
from datetime import datetime
import plotly.figure_factory as ff
import dash_table
from statistics import mean
import os

df_1=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run1\201210_run1_analysis_part1.xls',sheet_name=0)
df_2=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run2\201210_run2_analysis_part1.xls',sheet_name=0)
df_3=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run3\201204_run3_analysis_part1.xls',sheet_name=0)
df_4=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run4\201204_run4_analysis_part1.xls',sheet_name=0)
df_5=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run5\201204_run5_analysis_part1.xls',sheet_name=0)
df_6=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run6\201204_run6_analysis_part1.xls',sheet_name=0)
df_7=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run7\201204_run7_analysis_part1.xls',sheet_name=0)
df_8=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run8\201204_run8_analysis_part1.xls',sheet_name=0)
df_9=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run9\201204_run9_analysis_part1.xls',sheet_name=0)
df_10=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run10\201204_run10_analysis_part1.xls',sheet_name=0)
df_11=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run11\201204_run11_analysis_part1.xls',sheet_name=0)
df_12=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run12\201202_run12_analysis_part1.xls',sheet_name=0)
df_13=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run13\201202_run13_analysis_part1.xls',sheet_name=0)
df_14=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run14\201202_run14_analysis_part1.xls',sheet_name=0)
df_15=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run15\201202_run15_analysis_part1.xls',sheet_name=0)
df_16=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run16\201204_run16_analysis_part1.xls',sheet_name=0)
df_17=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run17\201202_run17_analysis_part1.xls',sheet_name=0)
df_18=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run18\201204_run18_analysis_part1.xls',sheet_name=0)

print('done')
Dataframes=[df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18]

Runs=['run_1','run_2','run_3','run_4','run_5','run_6','run_7','run_8','run_9','run_10','run_11','run_12','run_13','run_14','run_15','run_16','run_17']

df_1_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run1\201210_run1_analysis_part1.xls',sheet_name=1)
df_2_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run2\201210_run2_analysis_part1.xls',sheet_name=1)
df_3_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run3\201204_run3_analysis_part1.xls',sheet_name=1)
df_4_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run4\201204_run4_analysis_part1.xls',sheet_name=1)
df_5_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run5\201204_run5_analysis_part1.xls',sheet_name=1)
df_6_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run6\201204_run6_analysis_part1.xls',sheet_name=1)
df_7_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run7\201204_run7_analysis_part1.xls',sheet_name=1)
df_8_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run8\201204_run8_analysis_part1.xls',sheet_name=1)
df_9_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run9\201204_run9_analysis_part1.xls',sheet_name=1)
df_10_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run10\201204_run10_analysis_part1.xls',sheet_name=1)
df_11_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run11\201204_run11_analysis_part1.xls',sheet_name=1)
df_12_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run12\201202_run12_analysis_part1.xls',sheet_name=1)
df_13_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run13\201202_run13_analysis_part1.xls',sheet_name=1)
df_14_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run14\201202_run14_analysis_part1.xls',sheet_name=1)
df_15_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run15\201202_run15_analysis_part1.xls',sheet_name=1)
df_16_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run16\201204_run16_analysis_part1.xls',sheet_name=1)
df_17_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run17\201202_run17_analysis_part1.xls',sheet_name=1)
df_18_J=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run18\201204_run18_analysis_part1.xls',sheet_name=1)

Dataframes_Jis=[df_1_J,df_2_J,df_3_J,df_4_J,df_5_J,df_6_J,df_7_J,df_8_J,df_9_J,df_10_J,df_11_J,df_12_J,df_13_J,df_14_J,df_15_J,df_16_J,df_17_J,df_18_J]


df_1_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run1\201210_run1_analysis_part2.xlsx',sheet_name=-2)
df_2_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run2\201210_run2_analysis_part2.xlsx',sheet_name=-2)
df_3_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run3\201204_run3_analysis_part2.xlsx',sheet_name=-2)
df_4_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run4\201204_run4_analysis_part2.xlsx',sheet_name=-2)
df_5_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run5\201204_run5_analysis_part2.xlsx',sheet_name=-2)
df_6_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run6\201204_run6_analysis_part2.xlsx',sheet_name=-2)
df_7_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run7\201204_run7_analysis_part2.xlsx',sheet_name=-2)
df_8_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run8\201204_run8_analysis_part2.xlsx',sheet_name=-2)
df_9_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run9\201204_run9_analysis_part2.xlsx',sheet_name=-2)
df_10_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run10\201204_run10_analysis_part2.xlsx',sheet_name=-2)
df_11_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run11\201204_run11_analysis_part2.xlsx',sheet_name=-2)
df_12_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run12\201202_run12_analysis_part2.xlsx',sheet_name=-2)
df_13_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run13\201202_run13_analysis_part2.xlsx',sheet_name=-2)
df_14_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run14\201202_run14_analysis_part2.xlsx',sheet_name=-2)
df_15_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run15\201202_run15_analysis_part2.xlsx',sheet_name=-2)
df_16_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run16\201204_run16_analysis_part2.xlsx',sheet_name=-2)
df_17_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run17\201202_run17_analysis_part2.xlsx',sheet_name=-2)
df_18_T=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run18\201204_run18_analysis_part2.xlsx',sheet_name=-2)

Dataframes_Worker=[df_1_T,df_2_T,df_3_T,df_4_T,df_5_T,df_6_T,df_7_T,df_8_T,df_9_T,df_10_T,df_11_T,df_12_T,df_13_T,df_14_T,df_15_T,df_16_T,df_17_T,df_18_T]

df_1_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run1\201210_run1_analysis_part2.xlsx',sheet_name=-1)
df_2_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run2\201210_run2_analysis_part2.xlsx',sheet_name=-1)
df_3_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run3\201204_run3_analysis_part2.xlsx',sheet_name=-1)
df_4_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run4\201204_run4_analysis_part2.xlsx',sheet_name=-1)
df_5_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run5\201204_run5_analysis_part2.xlsx',sheet_name=-1)
df_6_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run6\201204_run6_analysis_part2.xlsx',sheet_name=-1)
df_7_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run7\201204_run7_analysis_part2.xlsx',sheet_name=-1)
df_8_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run8\201204_run8_analysis_part2.xlsx',sheet_name=-1)
df_9_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run9\201204_run9_analysis_part2.xlsx',sheet_name=-1)
df_10_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run10\201204_run10_analysis_part2.xlsx',sheet_name=-1)
df_11_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run11\201204_run11_analysis_part2.xlsx',sheet_name=-1)
df_12_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run12\201202_run12_analysis_part2.xlsx',sheet_name=-1)
df_13_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run13\201202_run13_analysis_part2.xlsx',sheet_name=-1)
df_14_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run14\201202_run14_analysis_part2.xlsx',sheet_name=-1)
df_15_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run15\201202_run15_analysis_part2.xlsx',sheet_name=-1)
df_16_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run16\201204_run16_analysis_part2.xlsx',sheet_name=-1)
df_17_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run17\201202_run17_analysis_part2.xlsx',sheet_name=-1)
df_18_s=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run18\201204_run18_analysis_part2.xlsx',sheet_name=-1)

Dataframes_Station=[df_1_s,df_2_s,df_3_s,df_4_s,df_5_s,df_6_s,df_7_s,df_8_s,df_9_s,df_10_s,df_11_s,df_12_s,df_13_s,df_14_s,df_15_s,df_16_s,df_17_s,df_18_s]

df_1_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run1\201210_run1_station_utilization_instation.xlsx')
df_2_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run2\201210_run2_station_utilization_instation.xlsx')
df_3_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run3\201204_run3_station_utilization_instation.xlsx')
df_4_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run4\201204_run4_station_utilization_instation.xlsx')
df_5_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run5\201204_run5_station_utilization_instation.xlsx')
df_6_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run6\201204_run6_station_utilization_instation.xlsx')
df_7_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run7\201204_run7_station_utilization_instation.xlsx')
df_8_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run8\201204_run8_station_utilization_instation.xlsx')
df_9_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run9\201204_run9_station_utilization_instation.xlsx')
df_10_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run10\201204_run10_station_utilization_instation.xlsx')
df_11_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run11\201204_run11_station_utilization_instation.xlsx')
df_12_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run12\201202_run12_station_utilization_instation.xlsx')
df_13_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run13\201202_run13_station_utilization_instation.xlsx')
df_14_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run14\201202_run14_station_utilization_instation.xlsx')
df_15_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run15\201202_run15_station_utilization_instation.xlsx')
df_16_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run16\201204_run16_station_utilization_instation.xlsx')
df_17_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run17\201202_run17_station_utilization_instation.xlsx')
df_18_i=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run18\201204_run18_station_utilization_instation.xlsx')

Dataframes_instation=[df_1_i,df_2_i,df_3_i,df_4_i,df_5_i,df_6_i,df_7_i,df_8_i,df_9_i,df_10_i,df_11_i,df_12_i,df_13_i,df_14_i,df_15_i,df_16_i,df_17_i,df_18_i]




df_1_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run1\201210_run1_station_utilization_working.xlsx')
df_2_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201210_run2\201210_run2_station_utilization_working.xlsx')
df_3_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run3\201204_run3_station_utilization_working.xlsx')
df_4_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run4\201204_run4_station_utilization_working.xlsx')
df_5_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run5\201204_run5_station_utilization_working.xlsx')
df_6_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run6\201204_run6_station_utilization_working.xlsx')
df_7_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run7\201204_run7_station_utilization_working.xlsx')
df_8_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run8\201204_run8_station_utilization_working.xlsx')
df_9_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run9\201204_run9_station_utilization_working.xlsx')
df_10_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run10\201204_run10_station_utilization_working.xlsx')
df_11_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run11\201204_run11_station_utilization_working.xlsx')
df_12_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run12\201202_run12_station_utilization_working.xlsx')
df_13_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run13\201202_run13_station_utilization_working.xlsx')
df_14_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run14\201202_run14_station_utilization_working.xlsx')
df_15_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run15\201202_run15_station_utilization_working.xlsx')
df_16_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run16\201204_run16_station_utilization_working.xlsx')
df_17_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201202_run17\201202_run17_station_utilization_working.xlsx')
df_18_iw=pd.read_excel(r'G:\Geteilte Ablagen\Arculus\03 Operations\01 Projects\Audi\2020_Audi_TVKLVM_Modulare_Produktion\06 Testing\03 Simulations\02 SIT_Mockup\201214_Datensatz_eval_2112\201204_run18\201204_run18_station_utilization_working.xlsx')

Dataframes_instation_working=[df_1_iw,df_2_iw,df_3_iw,df_4_iw,df_5_iw,df_6_iw,df_7_iw,df_8_iw,df_9_iw,df_10_iw,df_11_iw,df_12_iw,df_13_iw,df_14_iw,df_15_iw,df_16_iw,df_17_iw,df_18_iw]


dec=4
current_clicks=0


Runs=['run_1','run_2','run_3','run_4','run_5','run_6','run_7','run_8','run_9','run_10','run_11','run_12','run_13','run_14','run_15','run_16','run_17','run_18']
#%store -r Dataframes
#%store -r Dataframes_Jis
#%store -r Dataframes_Worker
#%store -r Dataframes_Station
#%store -r Dataframes_instation
#%store -r utilization_list 
#%store -r utilization_list_worker

#df=Dataframes_Worker[0]
dict_worker={'run','percentage working','percentage idle','percentage rotation','average rotations (1/h)'}
df_t = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')
#dict_worker['run'][0]=1

name_list=list()
average_list_working=list()
average_list_idle=list()
average_list_rotation=list()
average_list_rotations=list()
for i in range(len(Dataframes_Worker)):
    name_list.append('run_'+str(i+1))
    average_list_working.append(Dataframes_Worker[i]['percentage working'].mean())
    average_list_idle.append(Dataframes_Worker[i]['percentage idle'].mean())
    average_list_rotation.append(Dataframes_Worker[i]['percentage rotation'].mean())
    average_list_rotations.append(Dataframes_Worker[i]['average rotations (1/h)'].mean())
    
data={'run':name_list,
      'percentage working':average_list_working,
      'percentage idle':average_list_idle,
      'percentage rotation':average_list_rotation,
      'average rotations (1/h)':average_list_rotations}

df = pd.DataFrame(data)

df_dict=df.to_dict()

Dataframes[1]

#Function for the calculating the simult. Jis wagon
def simul(jis_df):
    start_time=int(jis_df['Jis-wagon opening'][0])
    end_time=int(jis_df['Jis-wagon completion'][len(jis_df)-1])
    simul=np.zeros(end_time-start_time)
    for i in range(len(jis_df)):
        start_jis=int(jis_df['Jis-wagon opening'][i]-start_time)
        end_jis=int(jis_df['Jis-wagon completion'][i]-start_time)
        for j in range(end_jis-start_jis):
            simul[start_jis+j]+=1
                
    simul_df = pd.DataFrame(simul,columns=['simul'])
    
    return simul_df


#function forcreating the pdf



#external_stylesheets = ['assets/style.css']

app = dash.Dash(__name__,title='arculs Modular Production')

#auth = dash_auth.BasicAuth(
 #   app,
#    VALID_USERNAME_PASSWORD_PAIRS
#)

all_comparison = {
    'product swirl': ['p-rate per worker (1/(min/worker))', 'p-rate hour (1/min)', 'p-rate hour average 10 (1/min)','p-rate (1/min)'],
    'Jis-wagon': ['Jis-wagon duration (min) - hist','Jis-wagon duration (min)','Jis-wagon development','time intervall (min) - hist','time intervall (min)','simultaneous Jis-wagon'],
    'utilization worker': ['worker percentage working', 'worker utilization'],
    'utilization station':['station percentage active','station percentage active working', 'station utilization']
}

run_options=[ {'label': 'run_1', 'value': 'run_1'},
                        {'label': 'run_2', 'value': 'run_2'},
                        {'label': 'run_3', 'value': 'run_3'},
                        {'label': 'run_4', 'value': 'run_4'},
                        {'label': 'run_5', 'value': 'run_5'},
                        {'label': 'run_6', 'value': 'run_6'},
                        {'label': 'run_7', 'value': 'run_7'},
                        {'label': 'run_8', 'value': 'run_8'},
                        {'label': 'run_9', 'value': 'run_9'},
                        {'label': 'run_10', 'value': 'run_10'},
                        {'label': 'run_11', 'value': 'run_11'},
                        {'label': 'run_12', 'value': 'run_12'},
                        {'label': 'run_13', 'value': 'run_13'},
                        {'label': 'run_14', 'value': 'run_14'},
                        {'label': 'run_15', 'value': 'run_15'},
                        {'label': 'run_16', 'value': 'run_16'},
                        {'label': 'run_17', 'value': 'run_17'},
                        {'label': 'run_18', 'value': 'run_18'}]



app.layout = html.Div(
    children=[html.Div(className='image', style={'backgroundColor': '#010A33','margin':0,'padding-bottom': 0},
                      children=[
                          html.Img(
                        src='arculus_1.png',
                          style={'padding-bottom': 0,'padding-left':100,'margin':0,'max-height':'150px','max-width':'200px','overflow': 'hidden'})
                      ])
                 ,
        html.Div(className='row',style={'backgroundColor': '#17828A','font-size':'20px'},
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.P(children='Select the run you want to analyse',
                                       style={'color':'#FFFFFF', 'padding-left':5,'padding-top':25}),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='run', options=run_options,
                                                      multi=True, value='run_1',
                                                      style={'backgroundColor': '#FFFFFF', 'margin-left':5},
                                                      className='run_selector'
                                                      ),
                                     ],
                                     ),
                                 html.P(children='Select the data you want to analyse Test',
                                       style={'color':'#FFFFFF', 'padding-left':5,'padding-top':25}),
                                 dcc.Dropdown(
                                        id='pre-comparison',
                                        options=[{'label': k, 'value': k} for k in all_comparison.keys()],
                                        value='product swirl',
                                        multi=False,style={'backgroundColor': '#FFFFFF', 'margin-left':5,'padding-left':-10}),
                                 html.P(children='Select the visualization',
                                       style={'color':'#FFFFFF', 'padding-left':5,'padding-top':25}),
                                dcc.Dropdown(
                                    id='comparison',
                                    value='p-rate per worker (1/(min/worker))',
                                    style={'backgroundColor': '#FFFFFF', 'margin-left':5,'padding-left':-10,'color':'#010A33'}),
                                 html.Button('Download', id='download', n_clicks=0, style={'backgroundColor': '#FFFFFF', 'margin-top':50, 'margin-left':5}),
                                 html.P(id='placeholder',style={'color':'#FFFFFF', 'padding-left':5,'padding-top':10})
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',style={'backgroundColor': '#FFFFFF','fontColor':'#FFFFF'},
                             children=[
                                 dcc.Graph(id='graph'),
                                 dash_table.DataTable(id='data-table',
                                                    columns=[{"name": i, "id": i, 'format': {"specifier": ".2f"}} for i in df.columns],
                                                    data=df.to_dict('records'), style_header={'backgroundColor': '#010A33'},
                                                                                    style_cell={
                                                                                        'backgroundColor': '#010A33',
                                                                                        'color': 'white',
                                                                                        'font-family':'IBM Plex Sans'
                                                                                        
                                                                                    },
                                                )

                             ])
                              ])
        ]

)

@app.callback(
    Output('run', 'multi'),
    Input('comparison','value'))
def set_multi(pre_comparison):
    if pre_comparison=='Jis-wagon development' or pre_comparison=='station utilization' or pre_comparison=='worker utilization':
        return False
    else:
        return True
@app.callback(
    Output('comparison','options'),
    Output('placeholder','children'),
    Input('pre-comparison','value'))
def set_comparison(pre_comparison):
    print(all_comparison[pre_comparison])
    return [{'label': i, 'value': i} for i in all_comparison[pre_comparison]],''




@app.callback(
    Output('graph','figure'),
    Input('run','value'),
    Input('comparison','value'),
    Input('pre-comparison','value'))
def update_graph(run,comp,pre_comp):
    if pre_comp=='product swirl':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=Dataframes[i]
                    line_name=Runs[i]
            print(line_name)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=df['finish index'],y=df[comp],mode='lines',name=line_name))
            fig.update_layout(
                xaxis_title='finish index',
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            return fig

        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(Dataframes[b])
                        label_list.append(Runs[b])


            print(len(df_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Scatter(x=df_list[i]['finish index'], y=df_list[i][comp],mode='lines',name=label_list[i]))
            fig.update_layout(
                xaxis_title='finish index',
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            return fig
        
    if comp=='Jis-wagon duration (min) - hist':
        df_list=list()
        value_list=list()
        if type(run)==str:
            fig=px.histogram(x=Dataframes_Jis[0]['Jis-wagon duration (min)'],nbins=45)

            
        
        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.extend(Dataframes_Jis[b]['Jis-wagon duration (min)'].values.tolist())
                        value_list.extend([Runs[b]]*len(Dataframes_Jis[i]['Jis-wagon duration (min)']))

            df=pd.DataFrame(dict(
                series=value_list,
                data=df_list))
        
        
            fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=45)
            
        fig.update_layout(
                xaxis_title='Jis-wagon duration (min)',
                yaxis_title='count',
        font_family='IBM Plex Sans')
            
            
    if comp=='Jis-wagon development':
        for i in range(len(Runs)):
            if run==Runs[i]:
                jis_df=Dataframes_Jis[i]
        fig = go.Figure(
            data=[
                go.Bar(
                name='Jis-wagon duration (min)',
                y=jis_df['Jis-wagon nr'],
                x=jis_df['Jis-wagon duration (min)'],
                offsetgroup=0,
                base=jis_df["sim time (min)"],
                orientation='h'
                ),
                go.Bar(name='Time Intervall',
                y=jis_df['Jis-wagon nr'],
                x=jis_df['time intervall (min)'],
                base=(jis_df['Jis-wagon duration (min)']+jis_df['sim time (min)']),
                offsetgroup=1,
                orientation='h'
                )
            ],
            layout=go.Layout(
                yaxis_title='Time in min',
                xaxis_title='JIS-wagon',
            font_family='IBM Plex Sans'))
        fig['layout']['yaxis']['autorange'] = "reversed"
        
        
        
    
    if comp=='time intervall (min) - hist':
        df_list=list()
        value_list=list()
        if type(run)==str:
            fig=px.histogram(x=Dataframes_Jis[0]['time intervall (min)'],nbins=65)

            
        
        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.extend(Dataframes_Jis[b]['time intervall (min)'].values.tolist())
                        value_list.extend([Runs[b]]*len(Dataframes_Jis[i]['time intervall (min)']))

            df=pd.DataFrame(dict(
                series=value_list,
                data=df_list))
        
        
            fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=65)
            
        fig.update_layout(
                xaxis_title='time intervall (min) - hist',
                yaxis_title='count',
        font_family='IBM Plex Sans')
            
    if comp=='time intervall (min)':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=Dataframes_Jis[i]
                    line_name=Runs[i]
            print(line_name)
            fig=go.Figure()
            fig.add_trace(go.Bar(x=df['Jis-wagon nr'],y=df['time intervall (min)'],name=line_name))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                xaxis_title='Jis-wagon nr',
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            
            return fig

        else:
            print('list')
            print(len(run))
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(Dataframes_Jis[b])
                        label_list.append(Runs[b])


            print(len(df_list))
            print(len(label_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['time intervall (min)'],name=label_list[i]))
            fig.update_layout(barmode='group')
            fig.update_layout(
                xaxis_title='Jis-wagon nr',
                yaxis_title=comp,
            font_family='IBM Plex Sans')

            return fig
        
    if comp=='Jis-wagon duration (min)':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=Dataframes_Jis[i]
                    line_name=Runs[i]
            print(line_name)
            fig=go.Figure()
            fig.add_trace(go.Bar(x=df['Jis-wagon nr'],y=df['Jis-wagon duration (min)'],name=line_name))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                xaxis_title='Jis-wagon nr',
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            
            return fig

        else:
            print('list')
            print(len(run))
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(Dataframes_Jis[b])
                        label_list.append(Runs[b])


            print(len(df_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['Jis-wagon duration (min)'],name=label_list[i]))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                xaxis_title='Jis-wagon nr',
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            
    
    if comp=='simultaneous Jis-wagon':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=simul(Dataframes_Jis[i])
                    line_name=Runs[i]
                    

                    
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=df['simul'],mode='lines',name=line_name))
            
            fig.update_layout(
                xaxis_title='time (sec)',
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            return fig

        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(simul(Dataframes_Jis[b]))
                        label_list.append(Runs[b])


            print(len(df_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Scatter(y=df_list[i]['simul'],mode='lines',name=label_list[i]))
                
            fig.update_layout(
                xaxis_title='time (sec)',
                yaxis_title=comp,
            font_family='IBM Plex Sans')

            return fig
        
        
    if comp=='station percentage active':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=Dataframes_Station[i]
                    line_name=Runs[i]
            fig=go.Figure()
            fig.add_trace(go.Bar(x=df['station name'],y=df['percentage active'],name=line_name))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            
            return fig

        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(Dataframes_Station[b])
                        label_list.append(Runs[b])


            print(len(df_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage active'],name=label_list[i]))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                yaxis_title=comp,
            font_family='IBM Plex Sans')

            return fig
        
        
    if comp=='station percentage active working':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=Dataframes_Station[i]
                    line_name=Runs[i]
            fig=go.Figure()
            fig.add_trace(go.Bar(x=df['station name'],y=df['percentage working active'],name=line_name))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            
            return fig

        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(Dataframes_Station[b])
                        label_list.append(Runs[b])


            print(len(df_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage working active'],name=label_list[i]))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                yaxis_title=comp,
            font_family='IBM Plex Sans')

            return fig
        
    if comp=='worker percentage working':
        df_list=list()
        label_list=list()
        if type(run)==str:
            for i in range(len(Runs)):
                if run==Runs[i]:
                    df=Dataframes_Worker[i]
                    line_name=Runs[i]
            fig=go.Figure()
            fig.add_trace(go.Bar(x=df['worker name'],y=df['percentage working'],name=line_name))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                yaxis_title=comp,
            font_family='IBM Plex Sans')
            
            return fig

        else:
            for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.append(Dataframes_Worker[b])
                        label_list.append(Runs[b])


            print(len(df_list))
            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Bar(x=df_list[i]['worker name'], y=df_list[i]['percentage working'],name=label_list[i]))
            fig.update_layout(barmode='group')
            
            fig.update_layout(
                yaxis_title=comp,
                font_family='IBM Plex Sans')

            return fig
        
    
    if comp=='station utilization':
        for i in range(len(Runs)):
            if run==Runs[i]:
                df=pd.DataFrame(utilization_list[i])
                
        fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True,
                      group_tasks=True)
        
        
        
        
    if comp=='worker utilization':
        for i in range(len(Runs)):
            if run==Runs[i]:
                df=pd.DataFrame(utilization_list_worker[i])
                
        fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True,
                      group_tasks=True)
        
        
    return fig  


@app.callback(
    [Output('data-table','columns'),
    Output('data-table','data')],
    [Input('run','value'),
    Input('pre-comparison','value')])
def update_data_table(run,pre_comp):
    if type(run)==str:
        run_list=list()
        run_list.append(run)
        run=run_list
    if pre_comp=='utilization worker':
        name_list=list()
        average_list_working=list()
        average_list_idle=list()
        average_list_rotation=list()
        average_list_rotations=list()
        print('utilization worker')
        for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        name_list.append(Runs[b])
                        average_list_working.append(round(Dataframes_Worker[b]['percentage working'].mean(),dec))
                        average_list_idle.append(round(Dataframes_Worker[b]['percentage idle'].mean(),dec))
                        average_list_rotation.append(round(Dataframes_Worker[b]['percentage rotation'].mean(),dec))
                        average_list_rotations.append(round(Dataframes_Worker[b]['average rotations (1/h)'].mean(),dec))
    
    
        data={'run':name_list,
          'percentage working':average_list_working,
          'percentage idle':average_list_idle,
          'percentage rotation':average_list_rotation,
          'average rotations (1/h)':average_list_rotations}
    


    
    if pre_comp=='product swirl':
        name_list=list()
        average_list_prate=list()
        average_list_pratehour=list()
        average_list_pratehour_init=list()
        average_list_prateworker=list()
        for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        name_list.append(Runs[b])
                        average_list_prate.append(round(Dataframes[b]['p-rate (1/min)'].mean(),dec))
                        average_list_pratehour.append(round(Dataframes[b]['p-rate hour (1/min)'].mean(),dec))
                        average_list_pratehour_init.append(round(Dataframes[b]['p-rate hour (1/min)'][200:-1].mean(),dec))
                        average_list_prateworker.append(round(Dataframes[b]['p-rate per worker (1/(min/worker))'].mean(),dec))
    
    
        data={'run':name_list,
          'p-rate':average_list_prate,
          'p-rate hour':average_list_pratehour,
          'p-rate hour after 200 jobs':average_list_pratehour_init,
          'p-rate per worker': average_list_prateworker}
        
        
    if pre_comp=='Jis-wagon':
        name_list=list()
        average_list_duration=list()
        average_list_timeintervall=list()
        average_list_simul=list()
        for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        name_list.append(Runs[b])
                        average_list_duration.append(round(Dataframes_Jis[b]['Jis-wagon duration (min)'].mean(),dec))
                        average_list_timeintervall.append(round(Dataframes_Jis[b]['time intervall (min)'].mean(),dec))
                        average_list_simul.append(round(simul(Dataframes_Jis[b]).mean(),dec))
    
    
        data={'run':name_list,
          'Jis-wagon duration (min)':average_list_duration,
          'time intervall (min)':average_list_timeintervall,
          'simultaneous Jis-wagon':average_list_simul}
        
        
    if pre_comp=='utilization station':
        name_list=list()
        average_list_active=list()
        average_list_working=list()
        for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        name_list.append(Runs[b])
                        average_list_active.append(round(Dataframes_Station[b]['percentage active'].mean(),dec))
                        average_list_working.append(round(Dataframes_Station[b]['percentage working active'].mean(),dec))
    
    
        data={'run':name_list,
          'percentage active':average_list_active,
          'percentage working active':average_list_working}
    

    df = pd.DataFrame(data)
        
    columns=[{"name": i, "id": i} 
                 for i in df.columns]
    
    data=df.to_dict('records')
    
    return columns,data


@app.callback(
    Output('placeholder','children'),
    Input('run','value'),
    Input('pre-comparison','value'),
    Input('download','n_clicks'))

def create_pdf(run,pre_comp,n_clicks):
    if n_clicks>0:
        if type(run)==str:
            run_list=list()
            run_list.append(run)
            run=run_list

        import os
        if not os.path.exists('fig_folder'):
            os.makedirs('fig_folder')


        #Initialize figure list   
        figure_list=list()

        figures=['p-rate per worker (1/(min/worker))', 'p-rate hour (1/min)', 'p-rate hour average 10 (1/min)','p-rate (1/min)']
        texts=['p-rateworker', 'p-ratehour', 'p-rateaverage10','p-rate']
        rounds=0
        for figure in figures:
            df_list=list()
            label_list=list()
            for i in range(len(run)):
                    for b in range(len(Runs)):
                        if run[i]==Runs[b]:
                            df_list.append(Dataframes[b])
                            label_list.append(Runs[b])


            fig = go.Figure()
            for i in range(len(label_list)):
                fig.add_trace(go.Scatter(x=df_list[i]['finish index'], y=df_list[i][figure],mode='lines',name=label_list[i]))
            fig.update_layout(
                xaxis_title='finish index',
                yaxis_title=figure,
                font_family='IBM Plex Sans')

            fig.write_html("fig_folder/"+texts[rounds]+".html")
            rounds+=1
            
        df_list=list()
        value_list=list()   
        for i in range(len(run)):
                for b in range(len(Runs)):
                    if run[i]==Runs[b]:
                        df_list.extend(Dataframes_Jis[b]['Jis-wagon duration (min)'].values.tolist())
                        value_list.extend([Runs[b]]*len(Dataframes_Jis[i]['Jis-wagon duration (min)']))

                df=pd.DataFrame(dict(
                series=value_list,
                data=df_list))
        
        
        fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=45)
            
        fig.update_layout(
                xaxis_title='Jis-wagon duration (min)',
                yaxis_title='count',
        font_family='IBM Plex Sans')
        
        fig.write_html("fig_folder/Jis-duration-hist.html")
        
        
        df_list=list()
        value_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.extend(Dataframes_Jis[b]['time intervall (min)'].values.tolist())
                    value_list.extend([Runs[b]]*len(Dataframes_Jis[i]['time intervall (min)']))

        df=pd.DataFrame(dict(
                series=value_list,
                data=df_list))
        
        
        fig=px.histogram(df,x='data',color='series',barmode='overlay',nbins=65)
            
        fig.update_layout(
                xaxis_title='time intervall (min) - hist',
                yaxis_title='count',
        font_family='IBM Plex Sans')
        
        fig.write_html("fig_folder/time-intervall-hist.html")
        
        
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.append(Dataframes_Jis[b])
                    label_list.append(Runs[b])
        fig = go.Figure()
        for i in range(len(label_list)):
            fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['time intervall (min)'],name=label_list[i]))
        fig.update_layout(barmode='group')
        fig.update_layout(
            xaxis_title='Jis-wagon nr',
            yaxis_title='time-intervall (min)',
            font_family='IBM Plex Sans')
        
        
        
        fig.write_html("fig_folder/time-intervall.html")
        
        
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.append(Dataframes_Jis[b])
                    label_list.append(Runs[b])


        fig = go.Figure()
        for i in range(len(label_list)):
            fig.add_trace(go.Bar(x=df_list[0]['Jis-wagon nr'], y=df_list[i]['Jis-wagon duration (min)'],name=label_list[i]))
        fig.update_layout(barmode='group')
            
        fig.update_layout(
                xaxis_title='Jis-wagon nr',
                yaxis_title='Jis-wagon duration (min)',
            font_family='IBM Plex Sans')
        
        
        fig.write_html("fig_folder/Jis-duration.html")
        
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.append(simul(Dataframes_Jis[b]))
                    label_list.append(Runs[b])


        fig = go.Figure()
        for i in range(len(label_list)):
            fig.add_trace(go.Scatter(y=df_list[i]['simul'],mode='lines',name=label_list[i]))
                
        fig.update_layout(
                xaxis_title='time (sec)',
                yaxis_title='simultenueous Jis-wagon',
            font_family='IBM Plex Sans')

        fig.write_html("fig_folder/simul.html")
        
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.append(Dataframes_Station[b])
                    label_list.append(Runs[b])

        fig = go.Figure()
        for i in range(len(label_list)):
            fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage active'],name=label_list[i]))
        fig.update_layout(barmode='group')
            
        fig.update_layout(
                yaxis_title='station percentage active',
            font_family='IBM Plex Sans')
        
        fig.write_html("fig_folder/station-active.html")
        
        
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.append(Dataframes_Station[b])
                    label_list.append(Runs[b])

        fig = go.Figure()
        for i in range(len(label_list)):
            fig.add_trace(go.Bar(x=df_list[i]['station name'], y=df_list[i]['percentage working active'],name=label_list[i]))
        fig.update_layout(barmode='group')
            
        fig.update_layout(
                yaxis_title='station percentage active working',
            font_family='IBM Plex Sans')
        
        fig.write_html("fig_folder/station-active-working.html")
        
        
        df_list=list()
        label_list=list()
        for i in range(len(run)):
            for b in range(len(Runs)):
                if run[i]==Runs[b]:
                    df_list.append(Dataframes_Worker[b])
                    label_list.append(Runs[b])


        fig = go.Figure()
        for i in range(len(label_list)):
            fig.add_trace(go.Bar(x=df_list[i]['worker name'], y=df_list[i]['percentage working'],name=label_list[i]))
        fig.update_layout(barmode='group')
            
        fig.update_layout(
                yaxis_title='worker percentage working',
                font_family='IBM Plex Sans')
        
        
        fig.write_html("fig_folder/worker-working.html")
        
        
        
        
        
        
        
        
        
            



        return('Success')
    
        
    
    

    
        
        
        
        
        
        
        


   

if __name__ == '__main__':
    app.run_server(debug=False)
