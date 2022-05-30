print('running')
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

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

%store Dataframes
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
%store Dataframes_Jis


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
%store Dataframes_Worker

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
%store Dataframes_Station

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


%store Dataframes_instation


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


%store Dataframes_instation_working
