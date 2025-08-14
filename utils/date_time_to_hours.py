

import numpy as np
from datetime import datetime


date_strings = ["2016-01-01 00:00:00", "2016-09-13 00:00:00", "2016-09-17 00:00:00", "2016-09-21 00:00:00", "2016-09-25 00:00:00", "2016-09-29 00:00:00", "2016-10-03 00:00:00", "2016-10-07 00:00:00"]

ics = []

for date_ in date_strings:
    date_obj = datetime.strptime(date_, '%Y-%m-%d %H:%M:%S') #datetime.fromisoformat(date_) 
    print(date_obj.timetuple())
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
    ics.append(int(hours_since_jan_01_epoch/6))
    print(day_of_year, hour_of_day)
    print("hours = ", hours_since_jan_01_epoch )
    print("steps = ", hours_since_jan_01_epoch/6) 


print(ics)

ics = []
for date_ in date_strings:
    date_obj = datetime.fromisoformat(date_) #datetime.strptime(date_, '%Y-%m-%d %H:%M:%S') #datetime.fromisoformat(date_) 
    print(date_obj.timetuple())
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
    ics.append(int(hours_since_jan_01_epoch/6))
    print(day_of_year, hour_of_day)
    print("hours = ", hours_since_jan_01_epoch )
    print("steps = ", hours_since_jan_01_epoch/6) 


print(ics)

