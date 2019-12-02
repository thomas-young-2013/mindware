def get_increasing_sequence(data):
    increasing_sequence = [data[0]]
    for item in data[1:]:
        _inc = increasing_sequence[-1] if item <= increasing_sequence[-1] else item
        increasing_sequence.append(_inc)
    return increasing_sequence
