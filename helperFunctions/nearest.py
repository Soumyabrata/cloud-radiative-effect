import numpy as np
import bisect



def nearest(ts,s):

	# Given a presorted list of timestamps:  s = sorted(index)
	i = bisect.bisect_left(s, ts)
	nearest_timestamp = min(s[max(0, i-1): i+2], key=lambda t: abs(ts - t))
	diff_timestamps = (nearest_timestamp - ts).total_seconds()
	return (nearest_timestamp,diff_timestamps)
	
	
	

