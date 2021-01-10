import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import datetime
import time
import default_config
import os
from timeit import default_timer as timer
from datetime import timedelta
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt

def start(neural_network):
    # Load data from csv file
    data = load_data_from_csv(default_config.data_path, default_config.data_filename)
    # Set time series index
    data = set_time_series_index(data)
    # Exploratory data analysis
    exploratory_data_analysis(data)
    # Data preprocessing
    data = data_preprocessing(data, neural_network)
    # Split data into train and test set and normalize it
    scaler, x_train, y_test, train_generator, test_generator = split_and_normalize_data(data, neural_network)
    # Create neural network structure model
    model = neural_network.create_model(data) 
    # Create early stopping to stop training at the point when performance on a test dataset starts to degrade
    early_stop = EarlyStopping(monitor='val_loss', patience=neural_network.patience, mode=neural_network.early_stop_mode)
    # Specify training configuration
    model.compile(optimizer=neural_network.optimizer,loss=neural_network.loss)

    trained_model_output = default_config.trained_model_path + "/" + default_config.train_set_folder
    trained = False
    # Check whether to train model or to load trained one
    if not (os.path.exists(trained_model_output + "/" + default_config.trained_model_name)):
        os.mkdir(trained_model_output)
        trained = True
        # Initialize the beginnin of training time
        start_training = timer()
        # Train the model
        model.fit_generator(train_generator,epochs=neural_network.train_epochs,validation_data=test_generator,callbacks=[early_stop])
        # Calculate training time
        end_training = timer()
        training_time = end_training - start_training
        training_time = timedelta(seconds=training_time)
        # Save trained model
        model.save(trained_model_output)
        # Plot lossed from trained model
        plot_losses(model)
        history = model.history
    else:
        # Load trained models
        model = tf.keras.models.load_model(default_config.trained_model_path +  "/" + default_config.train_set_folder)

    # Create predictions
    test_predictions, reversed_test_predictions = create_predictions(data, model, scaler, test_generator)
    # Save predictions plots
    save_prediction_plots(data, test_predictions, reversed_test_predictions)
    # Evaluate model on test data
    generator_evaluation, rmse, mae = evaluate_model_performance(model, neural_network.win_length, test_generator, y_test, test_predictions)
    # Do full data scaling to prepare data in order to forecast into the future
    full_scaler, full_generator = full_data_scaling(data, neural_network)
    # Create full model
    full_model = neural_network.create_model(data)
    # Compile full model
    full_model.compile(optimizer=neural_network.optimizer,loss=neural_network.loss)

    fully_trained_model_output = default_config.trained_model_path + "/" + default_config.full_data_set_folder

    # Check whether to train full model or to load trained one
    if not (os.path.exists(fully_trained_model_output + "/" + default_config.trained_model_name)):
        os.mkdir(fully_trained_model_output)
        # Train full model
        full_model.fit_generator(full_generator,epochs=neural_network.full_epochs)
        # Save trained model
        full_model.save(fully_trained_model_output)
    else:
        # Load fully trained model
        full_model = tf.keras.models.load_model(fully_trained_model_output)

    # Forecast into the future
    future_predictions = create_full_predictions(data, full_model, neural_network, full_scaler, y_test)
    # Save forecast plots
    save_future_predictions_plots(data, neural_network, future_predictions)
    # Evaluate full model performance
    full_generator_evaluation = model.evaluate_generator(full_generator, verbose=0)

    if (trained):
        # Create report
        finished_time = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        create_report(data, neural_network, training_time, finished_time, x_train, y_test, history, generator_evaluation, rmse, mae)

    return

def load_data_from_csv(data_path, filename):
    filename_path = data_path + "/" + filename
    if (os.path.exists(filename_path)):
        return pd.read_csv(filename_path)

def set_time_series_index(data):
    # Create index
    data['time'] = data.apply(lambda x : datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']), axis=1)
    # Set time as index
    data = data.set_index('time')

    return data

def exploratory_data_analysis(data):
    output = default_config.output_path + "/" + default_config.data_exploratory_folder

    if not (os.path.exists(output)):
        os.mkdir(output)

    # Heatmap - shows data correlation
    plt.figure(figsize=(16,6))
    plt.title('Heatmap - data correlation')
    sns.heatmap(data.corr(),cmap="YlGnBu",square=True,linewidths=.5,center=0,linecolor="red")
     # From the heatmap we can conclude that temperature and dew point are highly correlated with each other
    plt.savefig(output + "/" + "Heatmap1" + default_config.extension, bbox_inches = 'tight')

    # Clear the plot
    plt.clf()

    # Heatmap - shows null values
    plt.figure(figsize=(16,6))
    plt.title('Heatmap - null values')
    data.isnull().sum()
    sns.heatmap(data.isnull(),cmap="viridis")
    # From the heatmap we can conclude that data has null values for feature pm2.5 
    plt.savefig(output + "/" + "Heatmap2" + default_config.extension, bbox_inches = 'tight')

    # For each selected feature create lineplot
    for feature in data.iloc[:,[5,6,7,8]].columns:
        plt.clf()
        plt.figure(figsize=(16,6))
        plt.title('Lineplot - ' + feature)
        sns.lineplot(data=data,x=data.index,y=data[feature])
        plt.xticks(rotation=90)
        plt.savefig(output + "/" + "Lineplot - " + feature  + default_config.extension, bbox_inches = 'tight')

    # Lineplot - show monthly pm2.5 data
    plt.clf()   
    df_by_month = data.resample("M").sum()
    plt.figure(figsize=(16,6))
    plt.title('Lineplot - monthly - pm2.5')
    sns.lineplot(data=df_by_month,x=df_by_month.index,y=df_by_month['pm2.5'],color="red")
    plt.xticks(rotation=90)
    plt.savefig(output + "/" + "Lineplot - monthly - pm2.5" + default_config.extension, bbox_inches = 'tight')

    # Pointplot - show hourly pm2.5 data
    plt.clf()
    plt.figure(figsize=(16,6))
    plt.title('Pointplot - hourly - pm2.5')
    sns.pointplot(data=data,x=data.hour,y=data['pm2.5'],color="black")
    plt.savefig(output + "/" + "Pointplot - hourly - pm2.5" + default_config.extension, bbox_inches = 'tight')

    # Boxplot - show hourly pm2.5 data
    plt.clf()
    plt.figure(figsize=(16,6))
    plt.title('Boxplot - hourly - pm2.5')
    sns.boxplot(data=data,x=data['hour'],y=data['pm2.5'])
    plt.savefig(output + "/" + "Boxplot - hourly - pm2.5" + default_config.extension, bbox_inches = 'tight')

    # Boxplot - show monthly pm2.5 data
    plt.clf()
    plt.figure(figsize=(16,6))
    plt.title('Boxplot - monthly - pm2.5')
    sns.boxplot(data=data,x=data['hour'],y=data['pm2.5'])
    plt.savefig(output + "/" + "Boxplot - monthly - pm2.5" + default_config.extension, bbox_inches = 'tight')

    return

def data_preprocessing(data, neural_network):
    # Drop first 24 rows which have null pm2.5 value
    data = data.iloc[24:]
    # Drop last win_length rows so future forecasting can be done until the end of the year
    data = data.iloc[:-neural_network.cutoff_number]
    # Forward filling
    data = data.fillna(method='ffill')
    # Drop year, month, day, hour and registration number since these columns are not used for analysis
    data.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)
    # Delete further unused columns for forecasting
    del data['cbwd']
    del data['Iws']
    del data['Is']
    del data['Ir']

    # There are 4 features which are going to be used for forecasting: PM2.5, dew point, temperature, pressure and wind direction
    return data

def split_and_normalize_data(data, neural_network):
    # Create scaler for data normalization
    scaler = MinMaxScaler()
    # Normalize feature data (all columns)
    features = scaler.fit_transform(data)
    # Normalize target data - do not fit on target data
    target = scaler.transform(data)
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=neural_network.test_size, random_state=neural_network.random_state, shuffle=neural_network.shuffle)
    # Defune number of features used for forecasting
    num_features = features.shape[1]
    # Convert data to timeseries data
    # Time series data must be transformed into samples with input and output components
    train_generator = TimeseriesGenerator(x_train, y_train, length=neural_network.win_length ,sampling_rate = 1,batch_size=neural_network.batch_size)
    test_generator = TimeseriesGenerator(x_test, y_test, length=neural_network.win_length ,sampling_rate = 1,batch_size=neural_network.batch_size)

    return scaler, x_train, y_test, train_generator, test_generator

def plot_losses(model):
    # Determine how far the predicted values deviate from the actual values in the training dataset
    losses = pd.DataFrame(model.history.history)
    # Plot losses
    plt.figure(figsize=(16,6))
    plt.title('Train and Test losses')
    losses.plot()
    #plt.plot(model.history.history['loss'], label='train')
    #plt.plot(model.history.history['val_loss'], label='test')

    output = default_config.output_path + "/" + default_config.predictions_folder

    if not (os.path.exists(output)):
        os.mkdir(output)

    plt.savefig(output + "/" + "TrainAndTestLossesPlot" + default_config.extension, bbox_inches = 'tight')

def evaluate_model_performance(model, win_length, test_generator, y_test, predictions):
    generator_evaluation = model.evaluate_generator(test_generator, verbose=0)
    y_test = y_test[win_length:]
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, predictions))
     # Calculate  MAE
    mae = (mean_absolute_error(y_test, predictions))

    return generator_evaluation, rmse, mae

def create_predictions(data, model, scaler, test_generator):
    # Create predictions
    predictions = model.predict_generator(test_generator)
    # Inverse transformations
    reversed_predictions = scaler.inverse_transform(predictions)

    return predictions, reversed_predictions 

def save_prediction_plots(data, predictions, reversed_predictions):
    # Get predictions indexes(timestamp)
    index = data[predictions.shape[0]*-1:].index
    # Create dataframe out of reversed predictions
    rev_pred_df = pd.DataFrame(data=reversed_predictions, index = index, columns = data.columns)
    # Get a subset dataframe data which will be compared to calculated predictions
    df_test = data[predictions.shape[0]*-1:]
    # Create final dataframe which merges actual and predicted values
    df_final = pd.concat([df_test, rev_pred_df],axis=1)

    # For each selected feature create predictions plot
    for x in range(0, data.shape[1]):
        # Dataframe by feature
        df_feature = df_final[[data.columns[x]]]
        plt.clf()
        plt.figure(figsize=(16,6))
        plt.title('Predictions - ' + data.columns[x])
        plt.xlabel(data.index.name)
        plt.ylabel(data.columns[x])
        plt.plot(df_feature.iloc[:,[0]], label='Actual ' + data.columns[x] + ' value')
        plt.plot(df_feature.iloc[:,[1]], label='Predicted ' + data.columns[x] + ' value')
        plt.legend(loc="upper left")
        plt.savefig(default_config.output_path + "/" + default_config.predictions_folder + "/" + "Test Predictions" + data.columns[x] + default_config.extension, bbox_inches = 'tight')


def full_data_scaling(data, neural_network):
    # Create scaler for full data scaling
    full_scaler = MinMaxScaler()
    # Fit and transform data
    scaled_full_data = full_scaler.fit_transform(data)

    full_generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=neural_network.win_length ,sampling_rate = 1, batch_size=neural_network.batch_size)

    return full_scaler, full_generator

def create_full_predictions(data, full_model, neural_network, full_scaler, y_test):
    future_predictions = []

    # Take win_length size of data from the end of y_test dataset to use it for forecasting into the future
    first_eval_batch = y_test[-neural_network.win_length:]
    current_batch = first_eval_batch.reshape((1,neural_network.win_length,data.shape[1]))

    for i in range(neural_network.n_future):
        # Get prediction from current batch
        current_pred = full_model.predict(current_batch)
        # Store prediction
        future_predictions.append(current_pred[0])
        # Update batch to now include prediction and drop the first value which is no longer needed
        current_batch = np.append(current_batch[:,1:,:],[current_pred],axis=1)
    
    # Inverse transformation
    future_predictions = full_scaler.inverse_transform(future_predictions)

    return future_predictions

def save_future_predictions_plots(data, neural_network, future_predictions):

    plt.clf()
    plt.figure(figsize=(16,6))
    plt.title('Future Predictions')
    # Create forecast index (hourly)
    # Start index is the last timestamp index in original dataframe
    forecast_index = pd.date_range(start=data.iloc[[data.shape[0] - 1]].index[0],periods=neural_network.n_future,freq='H')
    # Create forecasted future dataframe
    df_future = pd.DataFrame(data=future_predictions,index=forecast_index,columns=data.columns)
 
    # Plot original data and forecast on the same axis
    ax = data.plot()
    df_future.plot(ax=ax)

    # Last timestamo index of data
    last_date = data.iloc[[data.shape[0] - 1]].index[0]
    # Get time index of the start date for plot graph
    start_date = last_date - datetime.timedelta(days=neural_network.zoom_days)
    # Get time index of the end date for plot graph - added days are equivavalnt to model win_length size
    end_date = last_date + datetime.timedelta(days=neural_network.win_length/24)
    # Zoom in the graph to see the forecast better
    plt.xlim(start_date,end_date)
    plt.savefig(default_config.output_path + "/" + default_config.predictions_folder + "/" + "Future Predictions" + default_config.extension, bbox_inches = 'tight')

    # Create dataframe for last couple of days until the end of the year
    # so direct comparison to the original graph could be made
    last_data = load_data_from_csv(default_config.data_path, default_config.data_filename)
     # Create index
    last_data['time'] = last_data.apply(lambda x : datetime.datetime(year=x['year'], month=x['month'], day=x['day'], hour=x['hour']), axis=1)
    # Set time as index
    last_data = last_data.set_index('time')
    # locate subset of original data where previos days before forecast and the forecast itself can be displayed
    last_data = last_data.iloc[-data.shape[0] - 1 - neural_network.win_length - neural_network.zoom_days:]
    last_data = last_data.fillna(method='ffill')
    # Drop year, month, day, hour and registration number since these columns are not used for analysis
    last_data.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)

    # Delete further unused columns for forecasting
    del last_data['cbwd']
    del last_data['Iws']
    del last_data['Is']
    del last_data['Ir']

    # Create another plot which displays future forecast
    # This is the reason why the original dataframe had a cutoff point at the end 
    # so the forecast could be drawn and merged to the original dataframe
    # and at the same time be comapred so the end result is visible in the plot
    plt.clf()
    plt.figure(figsize=(16,6))
    plt.title('Future Predictions Compared to Original')
    ax = data.plot()
    df_future.plot(ax=ax)
    # Zoom in the graph to see the forecast better
    plt.xlim(start_date,end_date)
    last_data.plot()
    plt.xlim(start_date,end_date)
    plt.savefig(default_config.output_path + "/" + default_config.predictions_folder + "/" + "OriginalGraphSubset" + default_config.extension, bbox_inches = 'tight')

def create_report(data, neural_network, training_time, finished_time, x_train, y_test, history, accuracy, rmse, mae):
    output_file = default_config.output_path + "/" + "Report - " + neural_network.name + ".log"

    with open(output_file, "w") as file:
        file.write("------------------- Report for {name} neural network-------------------\n".format(name=neural_network.name))
        file.write("\nNeural network parameters:\n")
        file.write("    Type of neural network:   {type}\n".format(type=neural_network.name))
        file.write("    Batch size:               {batch_size}\n".format(batch_size=neural_network.batch_size))
        file.write("    Sliding window length:    {win_length}\n".format(win_length=neural_network.win_length))
        file.write("    Number of epochs:         {epochs}\n".format(epochs=neural_network.train_epochs))
        file.write("    Early stopping patience:  {patience}\n".format(patience=neural_network.patience))
        file.write("    Total data samples:       {total}\n".format(total=data.shape[0]))
        file.write("    Number of features:       {num_features}\n".format(num_features=data.shape[1]))

        test_percentage = neural_network.test_size * 100
        train_percentage = 100 - test_percentage
        file.write("    Train set size:           {percentage}% ({samples} samples)\n".format(percentage=train_percentage, samples=x_train.shape[0]))
        file.write("    Test set size:            {percentage}% ({samples} samples)\n".format(percentage=test_percentage, samples=y_test.shape[0]))

        file.write("\nTraining Results:\n")
        for i in range(len(history.history["loss"])):
            file.write("    Epoch {epoch}: ".format(epoch=i+1))
            file.write("Loss: {loss}, ".format(loss=str(round(history.history["loss"][i], 4))))
            file.write("Validation loss: {loss}\n".format(loss=str(round(history.history["val_loss"][i], 4))))

        file.write("\nModel evaluation:\n")
        file.write("    Accuracy:                       {accuracy}\n".format(accuracy=str(round(accuracy, 5))))
        file.write("    RMSE (Root Mean Sqaure Error):  {rmse}\n".format(rmse=rmse))
        file.write("    MAE (Mean Square Error):        {mae}\n".format(mae=mae))

        file.write("\nModel time training:\n")
        file.write("    Training time:  {training_time}\n".format(training_time=training_time))
        file.write("    Finished time:  {finished_time}\n".format(finished_time=finished_time))
        