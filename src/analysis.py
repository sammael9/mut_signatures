import pandas as pd
import seaborn as sb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import paired_distances

iteration = 9500

#while iteration <= 5000:

signatures: pd.DataFrame = pd.read_csv(r'../resources/COSMIC_v3.2_SBS_GRCh37.txt', delimiter='\t')
signatures = signatures.sort_values(by=['Mutation type', 'Trinucleotide'], ascending=[True, True])

#original: pd.DataFrame = pd.read_csv(r'../resources/WGS_PCAWG.96.csv', delimiter=',')
#original = original.sort_values(by= ['Trinucleotide', 'Mutation type'], ascending=[True, True])

signatures.drop(columns=['Mutation type', 'Trinucleotide'], inplace=True)

#if iteration % 9500 == 0:

#    data: pd.DataFrame = pd.read_csv(r'../resources/output_catalog_' + str(iteration) + '.csv', delimiter=',',
#                                     header=None)
#    result = pd.concat([data, signatures], axis=1)
 #   correlations = result.corr(method='pearson')

  #  slice = correlations.iloc[0:130, 0:130]

 #   s = slice.unstack()
 #   so = s.sort_values(kind="quicksort")
 #   res = so[-9:]

  #  plot = sb.heatmap(correlations.iloc[0:130, 0:130], vmin=-1.0, vmax=1.0)
  #  plt.savefig(r'../resources/' + str(iteration) + '.pdf')
  #  plt.clf()

    #iteration = iteration + 1

data: pd.DataFrame = pd.read_csv(r'../resources/output_catalog_2780.csv', delimiter=',', header=None)

similarities_1 = pd.DataFrame()
similarities_2 = pd.DataFrame()
similarities_3 = pd.DataFrame()

sig_columns = list(signatures)
dat_columns = list(data)

for column1 in sig_columns:
    column_data_cos = []
    for column2 in dat_columns:
        cosine_result = 1-spatial.distance.cosine(data[column2], signatures[column1])
        column_data_cos.append(cosine_result)
    similarities_1[column1] = column_data_cos

for column1 in sig_columns:
    column_data_cos = []
    for column2 in sig_columns:
        cosine_result = 1-spatial.distance.cosine(signatures[column2], signatures[column1])
        column_data_cos.append(cosine_result)
    similarities_2[column1] = column_data_cos

for column1 in dat_columns:
    column_data_cos = []
    for column2 in dat_columns:
        cosine_result = 1-spatial.distance.cosine(data[column2], data[column1])
        column_data_cos.append(cosine_result)
    similarities_3[column1] = column_data_cos

sb.set(font_scale=0.4)

plot1 = sb.heatmap(similarities_1, vmin=0.0, vmax=1.0, cmap="RdYlGn", xticklabels=sig_columns, yticklabels=dat_columns)
plt.savefig(r'../resources/cosine_similarity_heatmap.png', dpi=500)
plt.clf()

plot2 = sb.heatmap(similarities_2, vmin=0.0, vmax=1.0, cmap="RdYlGn", xticklabels=sig_columns, yticklabels=sig_columns)
plt.savefig(r'../resources/cosine_intersimilarity_cosmic_heatmap.png', dpi=500)
plt.clf()

plot3 = sb.heatmap(similarities_3, vmin=0.0, vmax=1.0, cmap="RdYlGn", xticklabels=dat_columns, yticklabels=dat_columns)
plt.savefig(r'../resources/cosine_intersimilarity_our_heatmap.png', dpi=500)
plt.clf()

#print(1-spatial.distance.cosine())

similarities_1.to_csv(r'../resources/analysis_cos_sim.csv')
similarities_2.to_csv(r'../resources/analysis_cos_cosmic_sim.csv')
similarities_3.to_csv(r'../resources/analysis_cos_our_sim.csv')

print("Bye")

