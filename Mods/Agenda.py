import datetime
import pandas as pd
import os 

current_time = datetime.datetime.now()
current_hour, current_minute = datetime.datetime.time(current_time).hour, datetime.datetime.time(current_time).minute
current_date = datetime.datetime.date(datetime.datetime.today())

agenda_worksheet = 'agenda.xlsx'
agenda = pd.read_excel(agenda_worksheet)

description, responsible, hour_agenda = [], [], []
for index, row in agenda.iterrows():
    date = datetime.datetime.date(row['date'])
    complete_hour = datetime.datetime.strptime(str(row['hour']), '%H:%M:%S')
    hour = datetime.datetime.time(complete_hour).hour

    if current_date == date:
        if hour >= current_hour:
            description.append(row['description'])
            responsible.append(row['responsible'])
            hour_agenda.append(row['hour'])

def load_agenda():
    if description:
        return description, responsible, hour_agenda
    else:
        return False