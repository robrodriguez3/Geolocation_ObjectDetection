def dms_to_degrees(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    f_d = value[0]
    f_m = value[1]
    f_s = value[2]

    d=float(f_d)
    m=float(f_m)
    s=float(f_s)

    return d + (m / 60.0) + (s / 3600.0)
