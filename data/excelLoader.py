import pandas as pd
import os
import datetime
import data.dbmanager as db

ELLIA_base_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/EU_BG_EA'
NREL_base_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/EU_BG_EA'
# base_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/EU_BG_EA_W'
baseWind_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/EU_BG_EA/Wind/'
baseSolar_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/EU_BG_EA/Solar/'
baseTemp_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/EU_BG_EA/Temp/'

#  실제 구동 초기 명령어
def load_excel_files_to_db(opt):
    if opt is None:
        file_list = os.listdir(ELLIA_base_dir)
        base_dir = ELLIA_base_dir
    elif opt == "Wind":
        file_list = os.listdir(baseWind_dir)
        base_dir = baseWind_dir
    elif opt == "Solar":
        file_list = os.listdir(baseSolar_dir)
        base_dir = baseSolar_dir
    elif opt == "TempMax":
        read_weather_data_from_excel( baseTemp_dir,  "TempMax")
        return
    elif opt == "TempMin":
        read_weather_data_from_excel( baseTemp_dir,  "TempMin")
        return
    elif opt == "RadMin":
        read_weather_data_from_excel( baseTemp_dir,   "RadMin")
        return

    for filename in file_list:
        if filename.endswith('.xls'):
            print(filename)
            read_data_from_excel(base_dir, filename, opt)


# weather data from world weather
def read_weather_data_from_excel(base_dir, opt):
    # set directory with yours
    sheet_name = ""

    if opt is None:
        return
    elif opt == "TempMax":
        excel_file = "TX_STAID000017.xlsx"
        sheet_name = "TX_STAID000017"
    elif opt == "TempMin":
        excel_file = 'TN_STAID000017.xlsx'
        sheet_name = "TN_STAID000017"
    elif opt == "RadMin":
        excel_file = 'QQ_STAID000017.xlsx'
        sheet_name = "QQ_STAID000017"
    else:
        print("Undefined Job")
        return

    excel_dir = os.path.join(base_dir, excel_file)

    # read a excel file and make it as a DataFrame
    df_from_excel = pd.read_excel(excel_dir,  # write your directory here
                                  sheet_name=sheet_name,
                                  header=1,
                                  usecols='A,B,C,D',
                                  # names = ['region', 'sales_representative', 'sales_amount'],
                                  # dtype={'Measured & upscaled [MW]': float,
                                  #       'Monitored Capacity [MW]': float},
                                  #index_col=0,
                                  na_values='NaN',
                                  thousands=',',
                                  # nrows=10,
                                  comment='#')

    ndf = df_from_excel.to_numpy()

    conn = db.connectMariaDB()
    cursor = conn.cursor()

    for i in range(ndf.shape[0]):
        db_region_id = str(ndf[i][0])
        db_ymd = str(ndf[i][1])
        db_data = str(round(ndf[i][2], 2))
        db_data_qa = str(ndf[i][3])
        if opt == "TempMax":
            insert_sql = 'INSERT INTO AST0301 ( REGION,  REC_YMD, TEMP_MAX, DATA_QA ) VALUES ( ' \
                         + '\'' + db_region_id + '\', \'' \
                         + db_ymd + '\',  \'' \
                         + db_data + '\', \'' \
                         + db_data_qa + '\' )'
        elif opt == "TempMin":
            insert_sql = 'INSERT INTO AST0302 ( REGION,  REC_YMD, TEMP_MIN, DATA_QA ) VALUES ( ' \
                         + '\'' + db_region_id + '\', \'' \
                         + db_ymd + '\',  \'' \
                         + db_data + '\', \'' \
                         + db_data_qa + '\' )'
        elif opt == "RadMin":
            insert_sql = 'INSERT INTO AST0303 ( REGION,  REC_YMD, RAD_MIN, DATA_QA ) VALUES ( ' \
                         + '\'' + db_region_id + '\', \'' \
                         + db_ymd + '\',  \'' \
                         + db_data + '\', \'' \
                         + db_data_qa + '\' )'

        print(insert_sql)
        cursor.execute(insert_sql)

    conn.commit()
    db.closeMariaDB(conn)

    return df_from_excel


def read_NREL_data_from_excel(base_dir, filename):
    # set directory with yours
    if not filename:
        excel_file = 'WindForecast_20200801-20200831.xls'
    else:
        excel_file = filename

    excel_dir = os.path.join(base_dir, excel_file)

    # read a excel file and make it as a DataFrame
    df_from_excel = pd.read_excel(excel_dir,  # write your directory here
                                  sheet_name='forecast',
                                  header=4,
                                  usecols='A,E,F',
                                  # names = ['region', 'sales_representative', 'sales_amount'],
                                  # dtype={'Measured & upscaled [MW]': float,
                                  #       'Monitored Capacity [MW]': float},
                                  #index_col=0,
                                  na_values='NaN',
                                  thousands=',',
                                  # nrows=10,
                                  comment='#')

    df_from_excel.rename(columns={0: 'date', 1: 'power', 2: 'max'})
    ndf = df_from_excel.to_numpy()

    db_year =''
    db_month = ''
    db_day = ''
    db_hour = ''
    db_min = ''
    db_power = 0
    db_max = 0
    conn = db.connectMariaDB()
    cursor = conn.cursor()

    for i in range(ndf.shape[0]):
        time_obj = datetime.datetime.strptime(ndf[i][0], '%d/%m/%Y %H:%M')
        db_year = time_obj.strftime('%Y')
        db_month = time_obj.strftime('%m')
        db_day = time_obj.strftime('%d')
        db_hour = time_obj.strftime('%H')
        db_min = time_obj.strftime('%M')
        db_power = str(round(ndf[i][1], 2))
        db_max = str(round(ndf[i][2], 2))
        insert_sql = 'INSERT INTO AST0102 ( ' \
              '     REG_ID,' \
              '     REG_YMD,' \
              '     REG_HH24,' \
              '     REG_MM,' \
              '     REG_SS,' \
              '     PW_P,' \
              '     CAPACITY_MW,' \
              '     PRIM_REG_DT,' \
              '     LST_REG_DT ) VALUES  ( \'EU_BG_EA_W\', \'' + db_year + db_month + db_day + \
              '\', \'' + db_hour + '\', \'' + db_min + '\', ' + '\'00\',' +db_power + ',' + db_max + ', ' \
              'NOW(), NOW() )'

        check_sql = 'SELECT COUNT(*) AS YN FROM AST0102 WHERE REG_YMD= \'' + db_year + db_month + db_day +\
                    '\' AND REG_HH24 = \'' + db_hour + '\' AND REG_MM =\'' + db_min + '\''

        # date format coding check!
        cursor.execute(check_sql)
        rows = cursor.fetchall()
        # print(check_sql)
        cnt = int(rows[0][0])
        if cnt > 0:
            print("duplicate !")
        else:
            cursor.execute(insert_sql)
            # print(insert_sql)

    conn.commit()
    db.closeMariaDB(conn)

    return df_from_excel


# Belgium ELIA Company Data Excel
def read_data_from_excel(base_dir, filename, opt):
    # set directory with yours
    if filename is None:
        return
    else:
        excel_file = filename

    excel_dir = os.path.join(base_dir, excel_file)

    # read a excel file and make it as a DataFrame
    if opt == "Wind":
        df_from_excel = pd.read_excel(excel_dir,  # write your directory here
                                      sheet_name='forecast',
                                      header=4,
                                      usecols='A,E,F',
                                      # names = ['region', 'sales_representative', 'sales_amount'],
                                      # dtype={'Measured & upscaled [MW]': float,
                                      #       'Monitored Capacity [MW]': float},
                                      #index_col=0,
                                      na_values='NaN',
                                      thousands=',',
                                      # nrows=10,
                                      comment='#')
    elif opt == "Solar":
        df_from_excel = pd.read_excel(excel_dir,  # write your directory here
                                      sheet_name='SolarForecasts',
                                      header=4,
                                      usecols='A,F,G',
                                      # names = ['region', 'sales_representative', 'sales_amount'],
                                      # dtype={'Measured & upscaled [MW]': float,
                                      #       'Monitored Capacity [MW]': float},
                                      #index_col=0,
                                      na_values='NaN',
                                      thousands=',',
                                      # nrows=10,
                                      comment='#')

    df_from_excel.rename(columns={0: 'date', 1: 'power', 2: 'max'})
    ndf = df_from_excel.to_numpy()

    db_year =''
    db_month = ''
    db_day = ''
    db_hour = ''
    db_min = ''
    db_power = 0
    db_max = 0
    conn = db.connectMariaDB()
    cursor = conn.cursor()

    for i in range(ndf.shape[0]):
        time_obj = datetime.datetime.strptime(ndf[i][0], '%d/%m/%Y %H:%M')
        db_year = time_obj.strftime('%Y')
        db_month = time_obj.strftime('%m')
        db_day = time_obj.strftime('%d')
        db_hour = time_obj.strftime('%H')
        db_min = time_obj.strftime('%M')
        db_power = str(round(ndf[i][1], 2))
        db_max = str(round(ndf[i][2], 2))
        if opt == "Wind":
            insert_sql = 'INSERT INTO AST0102 ( ' \
                  '     REG_ID,' \
                  '     REG_YMD,' \
                  '     REG_HH24,' \
                  '     REG_MM,' \
                  '     REG_SS,' \
                  '     PW_P,' \
                  '     CAPACITY_MW,' \
                  '     PRIM_REG_DT,' \
                  '     LST_REG_DT ) VALUES  ( \'EU_BG_EA_W\', \'' + db_year + db_month + db_day + \
                  '\', \'' + db_hour + '\', \'' + db_min + '\', ' + '\'00\',' +db_power + ',' + db_max + ', ' \
                  'NOW(), NOW() )'

            check_sql = 'SELECT COUNT(*) AS YN FROM AST0102 WHERE REG_YMD= \'' + db_year + db_month + db_day +\
                        '\' AND REG_HH24 = \'' + db_hour + '\' AND REG_MM =\'' + db_min + '\''
        elif opt == "Solar":
            insert_sql = 'INSERT INTO AST0103 ( ' \
                  '     REG_ID,' \
                  '     REG_YMD,' \
                  '     REG_HH24,' \
                  '     REG_MM,' \
                  '     REG_SS,' \
                  '     PW_P,' \
                  '     CAPACITY_MW,' \
                  '     PRIM_REG_DT,' \
                  '     LST_REG_DT ) VALUES  ( \'EU_BG_EA_S\', \'' + db_year + db_month + db_day + \
                  '\', \'' + db_hour + '\', \'' + db_min + '\', ' + '\'00\',' +db_power + ',' + db_max + ', ' \
                  'NOW(), NOW() )'

            check_sql = 'SELECT COUNT(*) AS YN FROM AST0103 WHERE REG_YMD= \'' + db_year + db_month + db_day +\
                        '\' AND REG_HH24 = \'' + db_hour + '\' AND REG_MM =\'' + db_min + '\''
        else:
            db.closeMariaDB(conn)
            return

        # date format coding check!
        cursor.execute(check_sql)
        rows = cursor.fetchall()
        # print(check_sql)
        cnt = int(rows[0][0])
        cnt = 0
        if cnt > 0:
            print("duplicate !")
        else:
            # print(insert_sql)
            cursor.execute(insert_sql)
            # print(insert_sql)

    conn.commit()
    db.closeMariaDB(conn)

    return df_from_excel


if __name__ == '__main__':
    print('################ TESTING CODE #########################')
    load_excel_files_to_db("Wind")
