from datetime import datetime, timedelta, date as datedate
from pysolar.solar import get_altitude, get_azimuth
import pytz


def get_sunrise_sunset(centroid_lat, centroid_lon, date_str, tz='Asia/Hong_Kong'):
    """
    Compute approximate sunrise and sunset times using binary search on solar altitude.
    
    Parameters:
    - centroid_lat, centroid_lon: Coordinates of the region centroid.
    - date_str: Date in 'YYYY-MM-DD' format.
    - tz: Timezone string (default 'Asia/Hong_Kong').
    
    Returns:
    - tuple: (sunrise_time_str, sunset_time_str) in 'HH:MM' format.
    """
    d = datedate.fromisoformat(date_str)
    tz_obj = pytz.timezone(tz)
    
    def alt_time(h, m):
        dt = tz_obj.localize(datetime(d.year, d.month, d.day, h, m))
        utc_t = dt.astimezone(pytz.UTC)
        return get_altitude(centroid_lat, centroid_lon, utc_t)
    
    # Sunrise: find transition from negative to positive altitude (0:00 to 12:00)
    low, high = 0.0, 12.0
    for _ in range(30):  # High precision
        mid = (low + high) / 2
        ih = int(mid)
        im = int((mid - ih) * 60)
        a = alt_time(ih, im)
        if a < 0:
            low = mid
        else:
            high = mid
    sr_h = int(low)
    sr_m = int((low - sr_h) * 60)
    
    # Sunset: find transition from positive to negative altitude (12:00 to 24:00)
    low, high = 12.0, 24.0
    for _ in range(30):
        mid = (low + high) / 2
        ih = int(mid % 24)
        im = int((mid % 24 - ih) * 60)
        a = alt_time(ih, im)
        if a > 0:
            low = mid
        else:
            high = mid
    ss_h = int(low % 24)
    ss_m = int((low % 24 - ss_h) * 60)
    
    return f"{sr_h:02d}:{sr_m:02d}", f"{ss_h:02d}:{ss_m:02d}"
