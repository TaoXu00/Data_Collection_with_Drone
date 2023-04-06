import pandas as pd
import glob
import matplotlib.pyplot as plt
class Data_Preprocess():
    def load_data(self, file_dict, type):
        file_list = glob.glob(file_dict+"/*."+type)
        main_dataframe=pd.DataFrame(pd.read_table(file_list[0])).dropna()
        for i in range(1, len(file_list)):
            data_frame=pd.DataFrame(pd.read_table(file_list[i])).dropna()
            main_dataframe=pd.concat([main_dataframe,data_frame], axis=0)
        return main_dataframe

    def slice_dataframe_daily(self, Dataframe,features):
        new_df=Dataframe[features].copy()
        #print(new_df.columns.values.tolist())
        #new_df.info()
        new_df['Date']=pd.to_datetime(new_df['Date'])
        #new_df['year'] = new_df['Date'].dt.year
        new_df['year']=new_df['Date'].dt.year
        new_df['month'] = new_df['Date'].dt.month
        new_df['day']=new_df['Date'].dt.day
        #new_df.info()
        #print(new_df)
        return new_df

    def slice_data_frame_hourly(self, Dataframe, features, year, time_period, locations):
        df = Dataframe[features].copy()
        # print(new_df.columns.values.tolist())
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        #print(df)
        new_df=df.loc[(df['year']==year) & (df['month'].isin(time_period)) & (df['Location'].isin(locations))]
        print(new_df)
        #print(new_df)
        print(pd.unique(new_df['month']))
        print(pd.unique((new_df['Location'])))
        return new_df

    def statistics(self, df):
        locations= df['Location'].unique()
        years=df['year'].unique()
        dict = {}
        for year in years:
            dict[year]=[]

        for location in locations:
            #print(location)
            fig= plt.figure()
            year = df.loc[(df['Location'] == location) & (df['month'].isin([1,2,3,4]))].groupby(['year'])['VW_30cm'].count()
            for i in range(len(year.index)):
                if year.values[i]>2500:
                    dict[year.index[i]].append(location)
            max_sensor=0
            max_year=0
        for year in dict.keys():
            if len(dict[year])> max_sensor:
                max_sensor=len(dict[year])
                max_year=year
        print('%s: %s - %d' %(max_year, dict[max_year], len(dict[max_year])))
        for location in dict[max_year]:
            year_df = df.loc[(df['Location'] == location) & (df['year'] == max_year) & (df['month'].isin([1, 2, 3, 4]))]
            fig=plt.figure()
            count=year_df['VW_30cm'].count()
            plt.hist(year_df['VW_30cm'])
            #plt.plot(year_df['Date'], year_df['VW_30cm'])
            #plt.xticks(year['year'])
            #plt.savefig('plots/VW/Water Content distribution/%s_%s_%s.png' %(location, max_year, max_value))
            plt.savefig('plots/VW/Hourly/water content distribution Jan_April_2015/%s_%s_%s.png' % (location, max_year, count))
        return year_df

        '''
            month=year_df.groupby('month')['VM_30cm'].count()
            max_month=0
            max_value=0
            for i in range(len(month.index)):
                if month.values[i] > max_value:
                    max_value=month.values[i]
                    max_month=month.index[i]
            print('%s %s %d' %(location, max_month,max_value ))
            month_df = df.loc[(df['Location'] == location) & (df['year'] == max_year)&(df['month']==max_month)]
            fig = plt.figure()
            plt.hist(month_df['T_30cm'])
            # plt.xticks(year['year'])
            plt.savefig('plots/Temp/month temp distribution/%s_%s_%s_%s.png' % (location, max_year, max_month,max_value))
        '''
        '''
        for location in locations:
            print(location)
            fig = plt.figure()
            month = df.loc[(df['Location']==location) & (df['year'] == 2015)].groupby(['month'])['T_30cm'].count()
            # print(year)
            # year.columns
            plt.bar(month.index, month.values)
            # plt.xticks(year['year'])
            plt.savefig('plots/month_count_distribution/' + location + '.png')
        
        #these are the locations whose temperature distribution follows guassian.
        #locations=['CAF003', 'CAF019','CAF075', 'CAF095', 'CAF125','CAF141', 'CAF197', 'CAF215', 'CAF231', 'CAF237','CAF245','CAF308', 'CAF310','CAF314', 'CAF351', 'CAF357', 'CAF377', 'CAF397']
        #locations=df['Location'].tolist()
        # plot the data of 2025
        for location in locations:
            plt.figure()
            print(location)
            print(df.columns)
            location_df_2013 = df.loc[(df['Location'] == location) & (df['year'] == 2013)]
            location_df_2014= df.loc[(df['Location']==location) & (df['year'] == 2014)]
            location_df_2015=df.loc[(df['Location']==location) & (df['year'] == 2015)]
            #print(location_df)
            plt.plot(location_df_2013['Date'], location_df_2013['T_30cm'])
            plt.plot(location_df_2014['Date'],location_df_2014['T_30cm'])
            plt.plot(location_df_2015['Date'], location_df_2015['T_30cm'])
            plt.savefig('plots/Temperature_distribution/' + location + '2013_2014_2015.png')
        #df_2015=df.loc[df['year'] ==2015]
        #fig=plt.figure()
        #plt.plot()

        #print(year_list)

        date_list=[]
        
        '''
        '''
        for location in locations:
            date=df.loc[df['Location']==location]['Date'].unique()
            date_list.append(date)
        #print(date_list)
        '''

    def create_learning_dataframe(self, df):
        # selected the good dataset for learning, year=2015,
        locations = pd.unique(df['Location'])
        dates=pd.unique(df['Date'])
        times=pd.unique(df['Time'])
        # create a new dataframe
        num_found=0
        for date in dates:
            for time in times:
                num_place= df.loc[(df['Date']==date) & (df['Time'] ==time)][['Location']].count()[0]
                #print("num_place:%s" %(num_place))
                if num_place==len(locations):
                    num_found+=1
                    if num_found ==1:
                        locations = df.loc[(df['Date'] == date) & (df['Time'] == time)]['Location'].tolist()
                        cols = ['Date', 'Time'] + locations
                        new_df = pd.DataFrame(columns=cols)
                    date_value_df = df.loc[(df['Date'] == date) & (df['Time']==time)][['Location', 'VW_30cm']]
                    newrow=[date, time]
                    newrow=newrow+date_value_df['VW_30cm'].tolist()
                    new_df.loc[len(new_df.index)] =newrow
        print(new_df.info)
        print(new_df)
        new_df.to_csv('Dataset/water_content_2015_Jan_April_32_sensors')
        return new_df










