from datetime import timedelta


def end_of_month(dt):
    todays_month = dt.month
    tomorrows_month = (dt + timedelta(days=1)).month
    return tomorrows_month != todays_month