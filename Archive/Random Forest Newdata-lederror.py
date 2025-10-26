#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import hamming_loss
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
from matplotlib.legend_handler import HandlerLine2D
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from collections import defaultdict
from pycebox.ice import ice, ice_plot


# In[2]:


df = pd.read_csv('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/led_error.csv')


# In[3]:


column_ranges = [slice(0,5),slice(36,56)]
encoder = LabelEncoder()
for column_range in column_ranges:
    if isinstance(column_range, slice):
        selected_columns = df.iloc[:, column_range]
    else:
        selected_columns = df.iloc[:, column_range]

    for column in selected_columns.columns:
        df[column] = encoder.fit_transform(df[column])
df.to_csv('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/encodedfile_led_error.csv')
df


# In[4]:


scaler = MinMaxScaler()
column_numbers_to_scale = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                           30,31,32,33,34,35,36,37,38,3,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]

data_to_scale = df.values
data_to_scale[:, column_numbers_to_scale] = scaler.fit_transform(data_to_scale[:, column_numbers_to_scale])
df = pd.DataFrame(data_to_scale, columns = df.columns)
df = df.iloc[:,:].astype(float)

df.to_csv('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/scaledencodedfile.csv')


# In[5]:


df = df.iloc[:,:].fillna('-0.1')


# In[6]:


unused_columns = ['Error 2',
                  'Error 3','Error 1',
                  'Error 5','bms_istmodus_error']#'start_bms_soc_hires',
                  #'start_bms_nutzbarersoc','energy','laded_services_cdr_consumedenergy','avg_bms_istspannung',
                  #'end_bms_nutzbarersoc','avg_bms_iststrom_02','max_xbms_power_max'
                  #,'end_bms_soc_hires','avg_xbms_power_avg','duration_until_current_flow','avg_kl_istleistung',
                 #'start_bms_temperatur','start_bms_isttemperatur_hoechste','start_bms_isttemperatur_niedrigste',
                 #'end_bms_temperatur','end_bms_isttemperatur_niedrigste','alternating_stecker_status_counter',
                 #'max_hvlm_maxstrom_dcls','login_duration','end_bms_isttemperatur_hoechste','hvlm_ladeart']#,'avg_tme_verbrauch_gesamt'
                   #,'avg_tme_verbrauch_gesamt',
                  #'end_kbi_kilometerstand','start_kbi_kilometerstand','bms_kapazitaet','laded_hardware_enablertype']

df = df.drop(unused_columns, axis=1)


# In[7]:


error_columns=['Error 4']
X = df.drop(error_columns, axis=1)
y = df[error_columns]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


train_data = pd.concat([X_train, y_train], axis=1)

train_data.to_excel('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/train_data_led_error.xlsx',
                    index=False)

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_excel('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/test_data_led_error.xlsx',
                   index=False)


# In[10]:


n_estimators = [1,2,4,8,16,32,64,128]
train_results = []
test_results = []

for estimator in n_estimators:
    clf = RandomForestClassifier(n_estimators = estimator, n_jobs=-1, 
                                random_state = 20, oob_score = True, bootstrap = True, warm_start = True)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = clf.predict(X_test)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)


# In[11]:


line1, = plt.plot(n_estimators, train_results, 'b', label = 'train auc')
line2, = plt.plot(n_estimators, test_results, 'r', label = 'test auc')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.xlabel('n_estimators')
plt.ylabel('AUC Score')
plt.savefig('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/model_accuracy_led_error')
plt.show()


# In[12]:


clf = RandomForestClassifier(n_estimators = 32, n_jobs = -1, random_state = 15, oob_score = True,
                             bootstrap = True, max_depth=35)
clf.fit(X_train, y_train)
clf.oob_score_


# In[13]:


predictions = clf.predict(X_test)


# In[14]:


#for i, error_column in enumerate(error_columns):
#   column_predictions = predictions[:, i]
#    error_indices = [index for index, value in enumerate(column_predictions) if value == 1]
#    print(f"{error_column}: Rows with errors - {error_indices}")


# In[15]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[16]:


result = permutation_importance(
    clf, X_test, y_test, n_repeats=99, random_state=15, n_jobs=-1)
forest_importances = pd.Series(result.importances_mean, index=X_test.columns.tolist())
normalized_importances = (forest_importances - forest_importances.min()) / (forest_importances.max() - forest_importances.min())
normalized_importances = normalized_importances.sort_values(ascending=False)


# In[17]:


fig, ax = plt.subplots()
normalized_importances.plot(kind='bar', ax=ax)
ax.set_title("Feature Importances hvlm_zustand_led_error")
ax.set_ylabel("Normalized Importance Score")
ax.tick_params(axis='x', labelsize = 6)
fig.tight_layout()
plt.savefig('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/normalized_feature_importances_led_error')
plt.show()


# In[18]:


hamming_loss_value = hamming_loss(y_test, predictions)
print(f'Hamming Loss: {hamming_loss_value:.5f}')


# In[19]:


corr = spearmanr(X).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)


# In[20]:


distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))


# In[21]:


fig, ax1 = plt.subplots(figsize=(12,8))

dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
ax1.set_yticks(np.arange(0,2.7,0.25))
fig.tight_layout()
ax1.grid(True)
fig.tight_layout()
plt.savefig('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/dendrogram_led_error')
plt.show()

dendro_idx = np.arange(0, len(dendro["ivl"]))

#ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
#ax2.set_xticks(dendro_idx)
#ax2.set_yticks(dendro_idx)
#ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
#ax2.set_yticklabels(dendro["ivl"])
#_ = fig.tight_layout()


# In[22]:


cluster_ids = hierarchy.fcluster(dist_linkage, 0.80, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
additional_columns = ['charge_serviceprovidername','charge_platform','charge_partnername',
                     'charge_saleschannel','charge_cdr_cpo','charge_servicetype',
                     'charge_hardware_enablertype','chargingelectronics_version_brif','hvlm_charge_method','batterymanager_version_brif',
                     'country']
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features_names = X.columns[selected_features]

final_columns = list(selected_features_names)
final_columns.extend(additional_columns)

columns_to_drop = ['avg_bms_current_02','avg_bms_current','bms_capacity','country']

final_columns = [col for col in final_columns if col not in columns_to_drop]

X_train_sel = X_train[final_columns]
X_test_sel = X_test[final_columns]



# In[23]:


n_estimators_sel= [1,2,4,8,16,32,64,128]
train_results_sel = []
test_results_sel = []

for estimator in n_estimators:
    clf_sel = RandomForestClassifier(n_estimators = estimator, n_jobs=-1, 
                                random_state = 20, oob_score = True, bootstrap = True, warm_start = True)
    clf_sel.fit(X_train_sel, y_train)
    train_pred = clf_sel.predict(X_train_sel)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results_sel.append(roc_auc)
    y_pred = clf_sel.predict(X_test_sel)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results_sel.append(roc_auc)


# In[24]:


line3, = plt.plot(n_estimators_sel, train_results_sel, 'b', label = 'train auc')
line4, = plt.plot(n_estimators_sel, test_results_sel, 'r', label = 'test auc')
plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
plt.xlabel('n_estimators')
plt.ylabel('AUC Score')
plt.savefig('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/model_accuracy_colinear_led_error')
plt.show()


# In[25]:


clf_sel = RandomForestClassifier(n_estimators=32, random_state=42)
clf_sel.fit(X_train_sel, y_train)
print(
    "Baseline accuracy on test data with features removed:"
    f" {clf_sel.score(X_test_sel, y_test):.2}"
)


# In[26]:


result_sel = permutation_importance(
    clf_sel, X_test_sel, y_test, n_repeats=100, random_state=11, n_jobs=-1)
forest_importances_sel = pd.Series(result_sel.importances_mean, index=X_test_sel.columns.tolist())
normalized_importances_sel = (forest_importances_sel - forest_importances_sel.min()) / (forest_importances_sel.max() - forest_importances_sel.min())
normalized_importances_sel = normalized_importances_sel.sort_values(ascending=False)


# In[27]:


fig, ax = plt.subplots()
normalized_importances_sel.plot(kind='bar', ax=ax)
ax.set_title("Feature Importances Colinear error_detected_charging_continued")
ax.set_ylabel("Normalized Importance Score")
ax.tick_params(axis='x', labelsize = 6)
fig.tight_layout()
plt.savefig('C:/Users/arb/PycharmProjects/PCA_data/files/csv_files/random_forest_large/Colinear_features/normalized_feature_importances_colinear_led_error')
plt.show()


# In[28]:


feature_to_plot = 'laded_services_cdr_cpo'

# Create the partial dependence plots
partial_dependence_results = partial_dependence(clf_sel, X_train_sel, features=[feature_to_plot], grid_resolution=50)

# Extract the values and partial dependence values
x_values = partial_dependence_results['values'][0]
pd_values = partial_dependence_results['average'][0]

# Plot the partial dependence plot
plt.plot(x_values, pd_values)
plt.xlabel(feature_to_plot)
plt.ylabel('Partial Dependence')
plt.title(f'Partial Dependence Plot for {feature_to_plot}')
plt.show()


# In[ ]:


X_train_sel


# In[ ]:


tree_to_visualize = 0

plt.figure(figsize=(20, 10))
tree.plot_tree(clf_sel.estimators_[tree_to_visualize], filled=True,
               feature_names=X_test_sel.columns, class_names=True, rounded=True)
plt.show()


# In[ ]:


print(X_train_sel)


# In[ ]:




