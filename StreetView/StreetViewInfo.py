import GoogleSV as GSV

def GetLocByID(ID):
    return GSV.GetLocByID(panoid = ID)

def GetIDByLoc(latlon):
#
#   latlon is ['24', '121'] formt
#
    lat = str(latlon[0])
    lon = str(latlon[1])
    try:
        panoid = GSV.GetIDByLoc(lat = lat, lon = lon)
    except:
        panoid = None
    return panoid
