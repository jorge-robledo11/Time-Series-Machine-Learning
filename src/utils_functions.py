# Funciones de utilidad
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from matplotlib import gridspec
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import bartlett
from sklearn import metrics
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


def time_series_behavior(data: pd.DataFrame, kind='lineplot', series_estacionarias=None):
    
    """
    Visualiza series de tiempo, distribuciones de variables o cualquier otro tipo de gráfico en un DataFrame.

    Parameters:
        data (pd.DataFrame): El DataFrame con los datos a visualizar.
        kind (str): Tipo de gráfico a generar ('lineplot', 'histplot' u otros).

    Returns:
        None
    """
    
    if series_estacionarias is None:
        
        # Graficar las series de tiempo
        if kind == 'lineplot':
            for serie in data.columns:
                plt.figure(figsize=(18, 4))
                data[serie].plot(marker='.', markerfacecolor='springgreen', markersize=7.5, color='royalblue')
                plt.title(f'{serie}\n')
                plt.ylabel(serie)
                
                # Calcular automáticamente los límites de la escala para la serie
                min_value = data[serie].min()
                max_value = data[serie].max()
                plt.ylim(min_value, max_value)
                
                min_value = data[serie].index.min()
                max_value = data[serie].index.max()
                plt.xlim(min_value, max_value)
                
                plt.legend()
                plt.grid(color='white', linestyle='-', linewidth=0.25)
                plt.show()
        
        # Distribución de las variables        
        elif kind == 'histplot':
            for serie in data.columns:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=data, x=serie, bins=20, kde=True, lw=0.5, color='orange')
                plt.title(f'{serie}\n')
                plt.xlabel('')
                plt.grid(color='white', linestyle='-', linewidth=0.25)
                plt.tight_layout()
        
        else:
            print(f"Tipo de gráfico '{kind}' no es válido. Selecciona 'lineplot' o 'histplot'.")
    
    # Visualizar los cambios de estacionariedad en las series    
    else:
        for serie in data.columns:
            plt.figure(figsize=(18, 4))
            plt.plot(data.index, data[serie], label=serie, marker='.', color='royalblue', markerfacecolor='springgreen', markersize=7.5)
            plt.plot(series_estacionarias.index, series_estacionarias[serie], label=f'{serie} Estacionario', marker='.', markersize=7.5, 
                     color='aquamarine', markerfacecolor='fuchsia')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato de año (%Y)
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Mostrar un año por separado
            plt.title('Serie de Tiempo Original y Serie Estacionaria\n')
            plt.xlabel(data.index.name)
            plt.ylabel(serie)
            plt.grid(color='white', linestyle='-', linewidth=0.25)
            plt.legend()
            plt.tight_layout()
            plt.show()


def series_with_outliers(data:pd.DataFrame, threshold=3.5):
    
    # Creamos una copia de nuestro dataframe desastacionalizado
    df_ = data.copy()

    # Compute yhat using a rolling median and the rolling standard deviation which will be used as part of the threshold
    median_absolute_deviation = lambda y: np.median(np.abs(y - np.median(y)))

    for serie in df_.columns:
        rolling_stats = (df_[serie].rolling(window=12, center=True, min_periods=1).agg({f'rolling_median_{serie}': 'median', 
                                                                                        f'rolling_MAD_{serie}': median_absolute_deviation}))
        df_[[f'rolling_median_{serie}', f'rolling_MAD_{serie}']] = rolling_stats

        # Apply the threshold criteria to identify an outlier
        df_[f'is_outlier_{serie}'] = np.abs(df_[serie] - df_[f'rolling_median_{serie}']) > threshold * df_[f'rolling_MAD_{serie}']
        
    # Compute the upper and lower boundary of the threshold for plotting
    for serie in data.columns:
        df_[f'upper_{serie}'] = df_[f'rolling_median_{serie}'] + threshold * df_[f'rolling_MAD_{serie}']
        df_[f'lower_{serie}'] = df_[f'rolling_median_{serie}'] - threshold * df_[f'rolling_MAD_{serie}']

        # Plot
        fig, ax = plt.subplots(figsize=(18, 4))
        df_.plot(y=[serie], ax=ax, marker='.', markerfacecolor='springgreen', markersize=7.5, color='royalblue')
        df_.plot(y=[f'rolling_median_{serie}'], label=[f'MA {serie}'], ax=ax, linestyle='--', markersize=7.5, color='orange', legend=True)

        # If any data points are identified as outliers, plot them
        if df_[f'is_outlier_{serie}'].any():
            df_[serie].loc[df_[f'is_outlier_{serie}']].plot(marker='o', color='red', ax=ax, legend=None, linestyle='')
            
        ax.set_title(f'{serie} deseasonalised with outliers\n')
        ax.set_ylabel(f'{serie} deseasonalised')
        ax.grid(color='white', linestyle='-', linewidth=0.25)
        ax.set_xlabel(df_.index.name)
        
        # Escalas de las series
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max)

    return df_


def get_out_outliers(data_original:pd.DataFrame, data_transformada:pd.DataFrame, residuos:pd.DataFrame, methods:list):

    df_ = data_transformada.copy()
    
    for serie, m in zip(data_original.columns, methods):
        # Set outliers to NaN
        df_.loc[df_[f'is_outlier_{serie}'], serie] = np.nan

        # Apply linear interpolation
        df_.interpolate(method=m, inplace=True)

        # Add the seasonality extracted from STL back to deseaoned data
        df_[serie] = df_[serie] + residuos[serie]

        # Plot the data and location of the identified outliers from the rolling median method
        fig, ax = plt.subplots(nrows=2, figsize=[18, 6], sharex=True)
        data_original.plot(y=serie, marker='.', title='Before imputing outliers\n', ax=ax[0], color='royalblue',
                           markerfacecolor='springgreen', markersize=7.5)
        
        df_.plot(y=serie, marker='.', title='After imputing outliers\n', ax=ax[1], color='royalblue',
                 markerfacecolor='springgreen', markersize=7.5)
        
        if f'is_outlier_{serie}' in df_.columns and df_[f'is_outlier_{serie}'].any():
            df_[df_[f'is_outlier_{serie}']][serie].plot(marker='o', color='red', ax=ax[1], legend=None, linestyle='')

        # Escalas de las series
        y_min, y_max = ax[0].get_ylim()
        
        ax[0].set_ylabel(serie)
        ax[1].set_ylabel(serie)
        ax[0].set_xlabel(df_.index.name)
        ax[1].set_xlabel(df_.index.name)
        ax[0].set_ylim(y_min, y_max)
        ax[1].set_ylim(y_min, y_max)
        ax[0].grid(color='white', linestyle='-', linewidth=0.25)
        ax[1].grid(color='white', linestyle='-', linewidth=0.25)

    plt.show()
    
    # Las series necesarias
    series = [serie for serie in data_original.columns]
    return df_[series]


# Realizar la prueba de Dickey-Fuller
def adf_test(data:pd.DataFrame, significance_level=0.05):
    
    for var in data.columns:
        print(f'Serie: {var}')
        series = data[var].dropna()
        result = adfuller(series, autolag='AIC')
        print(f'P-value: {result[1]:0.3f}')

        if result[1] < significance_level:
            print(f'Se rechaza la hipótesis nula (la serie es estacionaria) a un nivel de significancia del {int(significance_level*100)}%\n')
        else:
            print(f'No se rechaza la hipótesis nula (la serie no es estacionaria) a un nivel de significancia del {int(significance_level*100)}%\n')
            
            
# Transformar series a estacionarias
def get_stationarity(data:pd.DataFrame, significance_level=0.05):
    
    stationary_dataframe = pd.DataFrame()  # DataFrame para las series estacionarias
    dif_dict = dict()  # Diccionario para el número de diferenciaciones
    
    for var in data.columns:
        series = data[var]
        dif = 0
        max_dif = 4  # Número máximo de diferenciaciones permitidas
        
        while dif <= max_dif:
            result = adfuller(series)
            if result[1] < significance_level:
                stationary_dataframe[var] = series  # Agregar la serie estacionaria al DataFrame
                dif_dict[var] = dif  # Agregar el número de diferenciaciones al diccionario
                break  # La serie es estacionaria, salimos del bucle
            else:
                series = series.diff().dropna()  # Realizar una diferenciación
                dif += 1
    
    return stationary_dataframe, dif_dict


# Prueba de Normalidad
def normality_test(data:pd.DataFrame, variables:list, significance_level=0.05):
    
    # Configurar figura
    fig = plt.figure(figsize=(22, 6))
    gs = gridspec.GridSpec(nrows=len(variables) // 3 + 1, ncols=3, figure=fig)

    for i, serie in enumerate(variables):
        ax = fig.add_subplot(gs[i // 3, i % 3])

        # Gráfico Q-Q
        stats.probplot(data[serie], dist='norm', plot=ax)
        ax.set_xlabel(serie)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels())
        ax.grid(color='white', linestyle='-', linewidth=0.25)

        # P-value
        p_value = stats.shapiro(data[serie])[1]
        ax.text(0.8, 0.9, f'p-value={p_value:0.3f}', transform=ax.transAxes, fontsize=13)

        # Imprime un mensaje de acuerdo a un umbral de significancia
        if p_value < significance_level:
            ax.text(0.44, 0.1, 'No se asemeja a una distribución normal', transform=ax.transAxes, fontsize=11, color='tomato')
        else:
            ax.text(0.48, 0.1, 'Se asemeja a una distribución normal', transform=ax.transAxes, fontsize=11, color='lightgreen')

    plt.tight_layout(pad=3)
    plt.show()
    
    
def homocedasticity_test(data:pd.DataFrame, significance_level=0.05):
    
    for serie in data.columns:
        print(f'Serie: {serie}')
        
        # Dividir la serie temporal en dos partes iguales
        n = len(data[serie])
        part1 = data[serie][:n//2]
        part2 = data[serie][n//2:]

        # Evaluar homocedasticidad con prueba de Bartlett
        bartlett_test = bartlett(part1, part2)

        if bartlett_test[1] > significance_level:
            print(f'No se rechaza la hipótesis nula de homocedasticidad (es homocedástica) a un nivel de significancia del {int(significance_level*100)}%\n')
        else:
            print(f'Se rechaza la hipótesis nula de homocedasticidad (no es homocedástica) a un nivel de significancia del {int(significance_level*100)}%\n')
       

# Gráfico de función de autocorrelación y autocorrelación parcial
def acf_y_pcf(data:pd.DataFrame, lags=40):

    # Generar el gráfico ACF
    for serie in data.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_acf(data[serie], lags=lags, ax=ax, color='crimson', zero=False, auto_ylims=True)
        plt.title(f'ACF para {serie}\n')
        plt.ylim(-1.1, 1.1)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()
    
    
    # Generar el gráfico de PACF
    for serie in data.columns:    
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_pacf(data[serie], lags=lags, ax=ax, color='green', zero=False, auto_ylims=True)
        plt.title(f'PACF para {serie}\n')
        plt.ylim(-1.1, 1.1)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()       
     
     
def plotting_train_test_pred(train_data:pd.Series, test_data:pd.Series, modelo=None, preds_data=None):
    
    # Graficamos los datos de entrenamiento y prueba
    if preds_data is not None:
        plt.figure(figsize=(18, 5))
        plt.plot(train_data.index, train_data, label='Train set', color='royalblue')
        plt.plot(test_data.index, test_data, label='Test set', color='lightgreen')
        plt.plot(preds_data.index, preds_data, label=type(modelo).__name__, color='orange')
        plt.title('Serie de Tiempo: Train set vs Test set vs Predicciones\n')
        plt.xlabel(train_data.index.name)
        plt.ylabel(train_data.name)
        plt.legend()
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()

    
    # Graficamos los datos de entrenamiento, prueba y las predicciones   
    else:
        plt.figure(figsize=(18, 5))
        plt.plot(train_data.index, train_data, label='Train set', color='royalblue')
        plt.plot(test_data.index, test_data, label='Test set', color='lightgreen')
        plt.title('Serie de Tiempo: Train set vs Test set\n')
        plt.xlabel(train_data.index.name)
        plt.ylabel(train_data.name)
        plt.legend()
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()
        

def evaluacion_metrica(y_true:pd.Series, y_pred:pd.Series, modelo, serie_nombre=None):
    
    metricas = dict()
    metricas['Modelo'] = type(modelo).__name__
    metricas['MSE'] = metrics.mean_squared_error(y_true, y_pred)
    metricas['RMSE'] = metrics.mean_squared_error(y_true, y_pred, squared=False)
    metricas['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
    metricas['MAPE'] = metrics.mean_absolute_percentage_error(y_true, y_pred)
    metricas['R²'] = metrics.r2_score(y_true, y_pred)
    
    return pd.DataFrame(metricas, index=[serie_nombre]).T


def plotting_residuals(modelo, series=None):
    
    # Crear una figura con ejes
    fig, ax = plt.subplots(figsize=(18, 4))

    # Graficar los residuos
    ax.plot(modelo.resid(), marker='.', color='royalblue', markerfacecolor='springgreen', markersize=7.5, label='Residuos')
    ax.set_ylabel('Residuos')
    ax.set_title('Gráfico de residuos en función del tiempo\n')
    ax.legend()
    ax.grid(color='white', linestyle='-', linewidth=0.25)

    # Mostrar la figura
    plt.tight_layout()
    plt.show()


def autocorrelation_test(data:pd.DataFrame, significance_level=0.05, lags=40):
    
    for serie in data.columns:
        print(f'Serie: {serie}')
        
        # Obtener los datos de la serie actual
        serie_data = data[serie]

        # Evaluar la autocorrelación entre los residuos
        pvalue = acorr_ljungbox(serie_data, lags=[lags], return_df=False)['lb_pvalue'].iloc[0]

        if pvalue < significance_level:
            print(f'Los residuos tienen autocorrelación a un nivel de significancia del {int(significance_level*100)}%\n')
        else:
            print(f'Los residuos no tienen autocorrelación a un nivel de significancia del {int(significance_level*100)}%\n')
            
            
def jarque_bera_test(data:pd.DataFrame, significance_level=0.05):
    
    for serie in data.columns:
        print(f'Serie: {serie}')
        
        # Obtener los datos de la serie actual
        serie_data = data[serie]

        # Evaluar si se parecen ruido blanco los residuos
        pvalue = jarque_bera(serie_data)[1]

        if pvalue < significance_level:
            print(f'Los residuos no parecen ruido blanco a un nivel de significancia del {int(significance_level*100)}%\n')
        else:
            print(f'Los residuos parecen ruido blanco a un nivel de significancia del {int(significance_level*100)}%\n')
            
            
def variance_stabilizers(serie: pd.Series):
    
    # Definir las transformaciones
    transformaciones = {'Log': np.log,
                        'Sqrt': np.sqrt,
                        'Reciprocal': np.reciprocal, 
                        'Quadratic': lambda x: x**2,
                        'Cubic': lambda x: x**3,
                        'Box-Cox': stats.boxcox(serie)}

    # Número de columnas para mostrar las transformaciones de dos en dos
    num_columnas = 2

    # Calcular el número de filas necesario
    num_transformaciones = len(transformaciones)
    num_filas = num_transformaciones // num_columnas
    if num_transformaciones % num_columnas != 0:
        num_filas += 1

    # Crear una figura con subtramas para mostrar las transformaciones de dos en dos
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(15, 5 * num_filas))

    # Iterar a través de las transformaciones y graficar de dos en dos
    for i, (nombre_transformacion, transformacion) in enumerate(transformaciones.items()):
        
        if nombre_transformacion != 'Box-Cox':
            transformed_data = serie.transform(transformacion)
        else:
            transformed_data = stats.boxcox(serie)[0]
        
        fila = i // num_columnas
        columna = i % num_columnas
        ax = axs[fila, columna]
        sns.distplot(transformed_data, hist=False, kde=True, color='royalblue', kde_kws={'shade': True, 'linewidth': 2}, ax=ax)
        ax.set_ylabel('')
        ax.grid(color='white', linestyle='-', linewidth=0.25)
        ax.set_title(f'{nombre_transformacion} Transformation\n')

    # Ocultar las subtramas vacías si no hay un número par de transformaciones
    if num_transformaciones % 2 != 0:
        axs[-1, -1].axis('off')

    plt.tight_layout()
    plt.show()
    

# RMSPE
def root_mean_square_percentage_error(y_true, y_pred) -> float:
    
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return rmspe


# Función de pérdida para calcular el rendimiento de los modelos
def loss_function(y_true, y_pred) -> tuple:
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    rmspe = root_mean_square_percentage_error(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    
    return mape, rmse, rmspe, r2


# Función para graficar los intervalos de confianza (Es necesario SKForecast)
def plotting_confidence_intervals_preds(test_data, preds_data, intervals_preds, modelo=None):
    
    plt.figure(figsize=(18, 5))
    plt.title('Intervalos de confianza: Test set vs Predicciones\n')
    plt.plot(test_data.index, test_data, label='Test set', color='lightgreen')
    plt.plot(preds_data.index, preds_data, label=type(modelo).__name__, color='orange', ls='--', marker='o', markersize=3.5)
    plt.fill_between(intervals_preds.index, intervals_preds['lower_bound'], intervals_preds['upper_bound'], 
                     color='xkcd:light purple', alpha=0.4, label='Intervalos de confianza [5, 95]')
    
    # Agregar anotaciones para los límites del intervalo de confianza con círculos rojos y contorno blanco
    for i in range(len(intervals_preds)):
        plt.scatter(intervals_preds.index[i], intervals_preds['lower_bound'][i], color='red', edgecolors='white', linewidth=0.75, zorder=5, s=35)
        plt.annotate(f'{intervals_preds["lower_bound"][i]:.2f}',
                     (intervals_preds.index[i], intervals_preds['lower_bound'][i]), 
                     textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8.5)
        
        plt.scatter(intervals_preds.index[i], intervals_preds['upper_bound'][i], color='red', edgecolors='white', linewidth=0.75, zorder=5, s=35)
        plt.annotate(f'{intervals_preds["upper_bound"][i]:.2f}', 
                     (intervals_preds.index[i], intervals_preds['upper_bound'][i]), 
                     textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8.5)
    
    plt.xlabel(test_data.index.name)
    plt.ylabel(test_data.name)
    plt.legend()
    plt.grid(color='white', linestyle='-', linewidth=0.25)
    plt.tight_layout()
    plt.show()
