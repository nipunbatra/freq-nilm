import sys
import psycopg2 as db
import pandas as pd
import datetime
import os
database_host = 'dataport.pecanstreet.org'
database_port = '5434'
database_name = 'postgres'
database_schema = 'university'
database_username='dqeRSAITtEVb'
database_password='GIr6e3WIXpI0'

conn = db.connect('host=' + database_host +
                          ' port=' + database_port +
                          ' dbname=' + database_name +
                          ' user=' + database_username +
                          ' password=' + database_password)

sql_query = """SELECT DISTINCT dataid from university.electricity_egauge_hours"""

list_of_buildings =  pd.read_sql(sql_query, conn).dataid.values

total = len(list_of_buildings)
for count, building_id in enumerate(list_of_buildings[:]):
	if os.path.exists("%d.csv" %int(building_id)):
		print(count, "/", total, building_id, "...Skipped... :)")
		continue
	print(count,"/",total, building_id, datetime.datetime.now())
	sql_query = """SELECT* FROM university.electricity_egauge_15min WHERE dataid=%d""" %int(building_id)
	df = pd.read_sql(sql_query, conn)
	df.to_csv("%d.csv" %int(building_id))