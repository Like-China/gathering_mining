from math import sin, asin, cos, radians, fabs, sqrt
EARTH_RADIUS = 6371  # 地球平均半径，6371km


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lng0, lat0, lng1, lat1):
    """用haversine公式计算球面两点间的距离。"""
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance*1000


if __name__ == "__main__":
    lng0, lat0, lng1, lat1 = 105.00231, -30.79532, 105.00231+0.001, -30.79532+0.001 # 公司到小区
    print(get_distance_hav(lng0, lat0, lng1, lat1))

