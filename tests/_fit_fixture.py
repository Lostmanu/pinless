"""A minimal FIT activity file, written in memory (no committed binary): a file_id, a sport and one record per point.

Only what the reader needs: the 14-byte header, one definition and its data messages per message type, and the CRC
of the whole file. Positions are semicircles, times seconds since the FIT epoch, altitude in fifths of a metre above
-500 m, as the FIT profile defines them.
"""
import datetime as dt
import struct

_CRC = (0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400)
EPOCH = dt.datetime(1989, 12, 31, tzinfo=dt.timezone.utc)
SPORTS = {"running": 1, "cycling": 2, "walking": 11, "hiking": 17}


def crc(data, c=0):
    for b in data:
        for nibble in (b & 0xF, b >> 4):
            tmp = _CRC[c & 0xF]
            c = (c >> 4) & 0x0FFF
            c ^= tmp ^ _CRC[nibble]
    return c


def _definition(local, global_, fields):
    out = struct.pack("<BBBHB", 0x40 | local, 0, 0, global_, len(fields))
    return out + b"".join(struct.pack("<BBB", num, size, base) for num, size, base in fields)


def write_fit(points, start, sport="hiking"):
    """points: [(lon, lat, seconds after start, metres)]; start: an aware datetime."""
    body = _definition(0, 0, [(0, 1, 0x00)]) + struct.pack("<BB", 0, 4)  # file_id: type activity
    body += _definition(1, 12, [(0, 1, 0x00)]) + struct.pack("<BB", 1, SPORTS[sport])  # sport
    body += _definition(2, 20, [(253, 4, 0x86), (0, 4, 0x85), (1, 4, 0x85), (2, 2, 0x84)])  # record
    t0 = int((start - EPOCH).total_seconds())
    for lon, lat, secs, metres in points:
        body += struct.pack("<BIiiH", 2, t0 + int(secs), round(lat * 2 ** 31 / 180), round(lon * 2 ** 31 / 180),
                            round((metres + 500) * 5))
    head = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    head += struct.pack("<H", crc(head))
    data = head + body
    return data + struct.pack("<H", crc(data))
