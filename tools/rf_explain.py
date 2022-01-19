import shap

data_csv = '/Users/jacaraba/Desktop/development/ilab/vietnam-lcluc/data/cloud_training_4band_rgb_fdi_si_ndwi.csv'
model_filename = '/Users/jacaraba/Desktop/development/ilab/vietnam-lcluc/data/cloud_training_4band_rgb_fdi_si_ndwi.pkl'

shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
