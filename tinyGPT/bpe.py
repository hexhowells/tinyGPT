def bytes_to_unicode() -> dict[int,str]:
    """
    Maps all bytes to unicode characters that can be rendered nicely

    Any bytes that are not rendered nicely are mapped to bytes past 256 (Ā onwards)

    Returns:
        dictionary mapping bytes to renderable unicode characters
    """
    nice_bytes = list(range(ord("!"), ord("~")+1)) + \
            list(range(ord("¡"), ord("¬")+1)) + \
            list(range(ord("®"), ord("ÿ")+1))
    all_bytes = nice_bytes[:]

    byte_idx = 0
    for byte in range(2**8):
        if byte not in nice_bytes:
            nice_bytes.append(byte)
            all_bytes.append(2**8 + byte_idx)  # convert ugly byte to next available byte
            byte_idx += 1
    
    all_bytes_chars = [chr(n) for n in all_bytes]  # map all bytes to unicode
    byte_to_char = dict(zip(nice_bytes, all_bytes_chars))

    return byte_to_char