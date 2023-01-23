# -*- coding: utf-8 -*-

import pandas as pd
from shapely.geometry import Point, shape

from flask import Flask
from flask import render_template
import json


data_path = './input/'


def get_tip_amount_segment(tip_amount):
    if tip_amount <= 1:
        return '0-1'
    elif tip_amount <= 2:
        return '1-2'
    elif tip_amount <= 3:
        return '2-3'
    elif tip_amount <= 4:
        return '3-4'
    elif tip_amount <= 5:
        return '4-5'
    elif tip_amount <= 10:
        return '5-10'
    # else:
    #     return '10+'

def get_total_amount_segment(total_amount):
    if total_amount <= 10:
        return '0-10'
    elif total_amount <= 20:
        return '10-20'
    elif total_amount <= 30:
        return '20-30'
    else:
        return '30+'

def get_passenger_count_segment(passenger_count):
    if passenger_count == 1:
        return '1'
    elif passenger_count == 2:
        return '2'
    elif passenger_count == 3:
        return '3'
    elif passenger_count == 4:
        return '4'
    else:
        return '5+'

# def get_location(longitude, latitude, provinces_json):
    
#     point = Point(longitude, latitude)

#     for record in provinces_json['features']:
#         polygon = shape(record['geometry'])
#         if polygon.contains(point):
#             return record['properties']['name']
#     return 'other'


# with open(data_path + '/geojson/china_provinces_en.json') as data_file:    
#     provinces_json = json.load(data_file)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    taxi_data = pd.read_csv(data_path + 'test.csv')
    nta_data = pd.read_csv(data_path + 'taxi-zone-lookup-with-ntacode.csv')
    # ev = pd.read_csv(data_path + 'events.csv')
    # ph_br_dev_model = pd.read_csv(data_path + 'phone_brand_device_model.csv')

    # df = gen_age_tr.merge(ev, how='left', on='device_id')
    # df = df.merge(ph_br_dev_model, how='left', on='device_id')
    # #Get n_samples records
    # df = df[df['longitude'] != 0].sample(n=n_samples)


    # top_10_brands_en = {'华为':'Huawei', '小米':'Xiaomi', '三星':'Samsung', 'vivo':'vivo', 'OPPO':'OPPO',
    #                     '魅族':'Meizu', '酷派':'Coolpad', '乐视':'LeEco', '联想':'Lenovo', 'HTC':'HTC'}

    # df['phone_brand_en'] = df['phone_brand'].apply(lambda phone_brand: top_10_brands_en[phone_brand] 
    #                                                 if (phone_brand in top_10_brands_en) else 'Other')

    

    # df['location'] = df.apply(lambda row: get_location(row['longitude'], row['latitude'], provinces_json), axis=1)
    df = taxi_data.merge(nta_data, how='left', left_on='pickup_location_id', right_on='location_id')
    df['passenger_count_segment'] = df['passenger_count'].apply(lambda passenger_count: get_passenger_count_segment(passenger_count))
    df['tip_amount_segment'] = df['tip_amount'].apply(lambda tip_amount: get_tip_amount_segment(tip_amount))
    df['total_amount_segment'] = df['total_amount'].apply(lambda total_amount: get_total_amount_segment(total_amount))

    cols_to_keep = ["id", "pickup_datetime", "dropoff_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_distance", "fare_amount", "tip_amount_segment","total_amount_segment", "pickup_location_id", "dropoff_location_id","borough", "zone","passenger_count_segment"]
    df_clean = df[cols_to_keep].dropna()
    df_clean = df_clean.rename(columns={'zone': 'pickup_zone'})
    df_clean = df_clean.rename(columns={'borough': 'pickup_borough'})

    

    df_clean = df_clean.merge(nta_data, how='left', left_on='dropoff_location_id', right_on='location_id')
    cols_to_keep = ["id", "pickup_datetime", "dropoff_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_distance", "fare_amount", "tip_amount_segment", "total_amount_segment", "pickup_location_id", "dropoff_location_id","pickup_zone","pickup_borough","borough","zone", "passenger_count_segment"]

    df_clean = df_clean[cols_to_keep].dropna()
    df_clean = df_clean.rename(columns={'zone': 'dropoff_zone'})
    df_clean = df_clean.rename(columns={'borough': 'dropoff_borough'})


    return df_clean.to_json(orient='records')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)